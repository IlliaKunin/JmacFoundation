import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from trafilatura import extract, fetch_url
from trafilatura.metadata import extract_metadata

from event_analyzer import analyze_event_people, analyze_events
from event_research import search_event_people
from news_search import search_news
from storage import (
    has_article, load_event_articles, load_event_people, load_events,
    mark_event_researched, normalize_title, save_event_article, save_event_people, save_events,
)


MAX_EVENT_SEARCH_ROUNDS = 5


def run_event_workflow(article_count=1, on_progress=None):
    if not isinstance(article_count, int) or not 1 <= article_count <= 20:
        raise ValueError("The number of new articles must be between 1 and 20.")
    load_dotenv(Path(__file__).with_name(".env"))
    report = on_progress or print
    history = load_event_articles()
    attempted_urls = set(history["url"])
    events = load_events()
    initial_events = len(events)
    people = load_event_people()
    initial_people = len(people)
    processed = 0
    for search_round in range(MAX_EVENT_SEARCH_ROUNDS):
        if processed >= article_count:
            break
        report(f"Searching US financial news from the last 14 days: {processed} of {article_count} new articles processed...")
        articles = search_news(
            max_results=article_count - processed, days=14,
            excluded_titles=history["article_title"], excluded_urls=attempted_urls,
            search_round=search_round,
        )
        for article in articles:
            if processed >= article_count:
                break
            url = article.get("url")
            if not url or url in attempted_urls or has_article(history, article):
                continue
            attempted_urls.add(url)
            report(f"Reading event article {processed + 1} of {article_count}: {article.get('title') or url}")
            text = article.get("raw_content")
            if not text:
                try:
                    html = fetch_url(url)
                    text = extract(html, include_links=True, include_comments=False) if html else None
                except Exception:
                    logging.exception("Could not read event source %s", url)
                    text = None
                if text:
                    try:
                        metadata = extract_metadata(html, default_url=url, extensive=False)
                        if metadata:
                            article = {**article, "published_date": article.get("published_date") or metadata.date}
                    except Exception:
                        logging.exception("Could not read event metadata for %s", url)
            if not text:
                report("Could not read this news source. Looking for a replacement...")
                continue
            report("Extracting the financial events, descriptions, and named participants...")
            extracted = analyze_events(article, text, on_progress=report)
            events = save_events(article, extracted)
            history = save_event_article(article)
            processed += 1

    pending = events.loc[events["research_complete"].ne("yes")].to_dict("records")
    for number, row in enumerate(pending, start=1):
        event = {**row, "participants": json.loads(row["participants"])}
        organizations = {}
        for participant in event["participants"]:
            if participant["entity_type"] == "organization":
                organizations.setdefault(normalize_title(participant["name"]), participant["name"])
        for organization in organizations.values():
            report(f"Event {number} of {len(pending)}: finding people connected through {organization}...")
            sources = search_event_people(event, organization)
            report(f"Checking names, event connections, and published contacts for {organization}...")
            identified = analyze_event_people(event, organization, sources, on_progress=report)
            people = save_event_people(event, identified)
        mark_event_researched(event["event_id"])
    people.attrs.update(
        requested_articles=article_count, processed_articles=processed,
        new_events=len(events) - initial_events, researched_events=len(pending),
        new_people=len(people) - initial_people,
    )
    return people
