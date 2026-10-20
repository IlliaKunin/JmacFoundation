import logging
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from trafilatura import extract, fetch_url
from trafilatura.metadata import extract_metadata

from deal_analyzer import analyze_news
from news_search import NEWS_SEARCH_DAYS, search_news
from seller_contacts import research_seller_contacts
from storage import (
    has_article, load_news_articles, load_news_history, save_news_article, save_news_decision,
    NEWS_SCREENING_VERSION, normalize_title,
)


MAX_NEWS_EMPTY_ROUNDS = 3


def _read_article(article):
    excerpt = article.get("content")
    excerpt = excerpt.strip() if isinstance(excerpt, str) else ""

    def with_excerpt(page_text):
        if not page_text or not page_text.strip():
            return excerpt
        if excerpt and excerpt != page_text.strip():
            return f"Search excerpt:\n{excerpt[:3000]}\n\nPage text:\n{page_text}"
        return page_text

    text = article.get("raw_content")
    if isinstance(text, str) and text.strip():
        return article, with_excerpt(text)
    try:
        html = fetch_url(article["url"])
        text = extract(html, include_comments=False) if html else None
    except Exception:
        logging.exception("Could not read news source %s", article["url"])
        html, text = None, None
    if text and not article.get("published_date"):
        try:
            metadata = extract_metadata(html, default_url=article["url"], extensive=False)
            if metadata:
                article = {**article, "published_date": metadata.date}
        except Exception:
            logging.exception("Could not read news metadata for %s", article["url"])
    # Keep the query-focused excerpt before the page text, including when a
    # blocked page returns navigation/cookie text instead of useful reporting.
    return article, with_excerpt(text if isinstance(text, str) else "")


def _current_history(history):
    return history.loc[
        history["screening_version"].eq(NEWS_SCREENING_VERSION)
        | history["decision"].isin(["saved", "pending_contacts"])
    ]


def run_news_workflow(article_count=1, on_progress=None):
    if type(article_count) is not int or not 1 <= article_count <= 20:
        raise ValueError("The number of new news articles must be between 1 and 20.")
    load_dotenv(Path(__file__).with_name(".env"))
    report = on_progress or print
    df = load_news_articles()
    history = _current_history(load_news_history())
    attempted_urls = set(df["url"]) | set(history["url"])
    processed, saved, excluded, rounds = 0, 0, 0, 0
    checked_sellers = 0

    def complete_article(article, analysis):
        nonlocal df, history, saved, checked_sellers
        # Journal the source analysis before contact calls. A retry can continue
        # from these participants without repeating article extraction or Groq.
        pending = {**analysis, "contact_research_pending": True}
        history = _current_history(save_news_decision(article, pending))
        sellers = pd.DataFrame([
            {
                "article_title": article.get("title") or "",
                "description": analysis["description"], "url": article["url"],
                "participant_name": person["name"], "participant_type": person.get("type", ""),
                "organization": person.get("organization") or (
                    person["name"] if person.get("type") not in {"individual", "family", ""} else ""
                ),
                "role": person.get("role", ""), "deal_side": "Seller",
            }
            for person in analysis["participants"]
            if person.get("deal_side") == "Seller" and normalize_title(person.get("name"))
        ])
        contacts = research_seller_contacts(sellers, on_progress=report)
        checked_sellers += contacts.attrs.get("checked_sellers", 0)
        df = save_news_article(article, analysis["description"], participants=analysis["participants"])
        history = _current_history(save_news_decision(article, analysis))
        if df.attrs.get("article_inserted", True):
            saved += 1
        else:
            report("Another run already saved this article. Looking for a replacement...")

    # Resume interrupted articles first; each completed article counts toward
    # this run's requested article total, rather than triggering extra discovery.
    for row in history.loc[history["decision"].eq("pending_contacts")].to_dict("records"):
        if saved >= article_count:
            break
        report(f"Resuming seller contacts: {row['article_title']}")
        pending_article = {"url": row["url"], "title": row["article_title"], "published_date": row["published_date"]}
        pending_analysis = {
            "qualifies": True, "description": row["description"], "reason": row["reason"],
            "us_scope": row["us_scope"], "us_reason": row["us_reason"],
            "participants": json.loads(row["participants"]),
        }
        complete_article(pending_article, pending_analysis)

    empty_rounds = 0
    while saved < article_count:
        search_round = rounds
        rounds += 1
        report(f"Finding news from the last {NEWS_SEARCH_DAYS} days: {saved} of {article_count} saved (search round {rounds})...")
        articles = search_news(
            max_results=article_count - saved, days=NEWS_SEARCH_DAYS,
            excluded_titles=[*df["article_title"], *history["article_title"]], excluded_urls=attempted_urls,
            search_round=search_round,
        )
        new_sources = 0
        for article in articles:
            if saved >= article_count:
                break
            url = article.get("url")
            if not url or url in attempted_urls or has_article(df, article) or has_article(history, article):
                continue
            attempted_urls.add(url)
            new_sources += 1
            report(f"Reading: {article.get('title') or url}")
            article, text = _read_article(article)
            if not text or not text.strip():
                report("No source text was available. Looking for a replacement...")
                continue
            report(f"Checking new article {processed + 1}; {saved} of {article_count} qualifying articles saved...")
            analysis = analyze_news(text, article.get("title") or "", on_progress=report)
            if analysis["qualifies"] and not any(
                person.get("deal_side") == "Seller" and normalize_title(person.get("name"))
                for person in analysis["participants"]
            ):
                analysis = {
                    **analysis, "qualifies": False,
                    "reason": "No named seller-side participants were identified in the source.",
                }
            if analysis["qualifies"]:
                complete_article(article, analysis)
            else:
                excluded += 1
                report(f"Excluded from the main table: {analysis['reason']}")
                history = _current_history(save_news_decision(article, analysis))
            processed += 1
            report(f"Saved {saved} of {article_count}; checked {processed} articles, excluded {excluded}.")
        empty_rounds = empty_rounds + 1 if new_sources == 0 else 0
        if empty_rounds >= MAX_NEWS_EMPTY_ROUNDS:
            report(f"No new sources found in {empty_rounds} consecutive searches. Saved {saved} of {article_count} requested articles.")
            break
    df.attrs.update(
        requested_articles=article_count, processed_articles=processed,
        saved_articles=saved, excluded_articles=excluded, search_rounds=rounds,
        checked_sellers=checked_sellers,
        stop_reason="target_reached" if saved >= article_count else "no_new_sources",
    )
    return df
