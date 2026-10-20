import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from trafilatura import fetch_url, extract
from trafilatura.metadata import extract_metadata

from donor_search import DONOR_QUERY, search_donors
from deal_analyzer import analyze_deal, analyze_funding
from donor_research import research_date, search_donor_details
from storage import has_article, load_donors, save_donors, unique_donor_names
from storage import load_processed_articles, save_processed_article
from storage import load_funding_research, normalize_title, save_funding_research


MAX_DISCOVERY_ROUNDS = 5


def run_donor_search(max_results=3, on_progress=None):
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("The number of new sources must be an integer between 1 and 20.")
    load_dotenv(Path(__file__).with_name(".env"))
    report = on_progress or print
    df = load_donors()
    initial_names = len(unique_donor_names(df))
    history = load_processed_articles()
    attempted_urls = set(history["url"])
    processed = 0
    rounds = 0
    for search_round in range(MAX_DISCOVERY_ROUNDS):
        if processed >= max_results:
            break
        rounds += 1
        report(f"Finding new sources: {processed} of {max_results} processed (search round {rounds})...")
        articles = search_donors(
            DONOR_QUERY,
            max_results=max_results - processed,
            excluded_titles=history["article_title"],
            excluded_urls=attempted_urls,
            search_round=search_round,
        )
        for article in articles:
            if processed >= max_results:
                break
            url = article.get("url")
            if not url or url in attempted_urls or has_article(history, article):
                continue
            attempted_urls.add(url)
            report(f"Reading new source {processed + 1} of {max_results}: {article.get('title') or url}")
            try:
                html = fetch_url(url)
                article_text = extract(
                    html,
                    output_format="txt",
                    include_comments=False,
                    include_links=True,
                    include_images=False,
                    include_formatting=False,
                ) if html else None
            except Exception:
                logging.exception("Could not read source %s", url)
                article_text = None
            if not article_text:
                report("Could not read this source. Looking for a replacement...")
                continue

            try:
                metadata = extract_metadata(html, default_url=url, extensive=False)
            except Exception:
                logging.exception("Could not read source metadata for %s", url)
                metadata = None
            if metadata:
                article = {
                    **article,
                    "published_date": article.get("published_date") or metadata.date,
                    "content": article.get("content") or metadata.description,
                }

            report(f"Finding all donors in new source {processed + 1} of {max_results}...")
            result = analyze_deal(article_text, article.get("title", ""), on_progress=report)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            df = save_donors(article, article_text, result)
            history = save_processed_article(article)
            processed += 1
            report(f"Processed {processed} of {max_results} new sources.")
    if processed < max_results:
        report(f"Found {processed} of {max_results} readable new sources after {rounds} search rounds.")
    df.attrs.update(
        requested_articles=max_results,
        processed_articles=processed,
        search_rounds=rounds,
        new_donors=len(unique_donor_names(df)) - initial_names,
    )
    return df


def run_donor_research(max_results=3, on_progress=None):
    load_dotenv(Path(__file__).with_name(".env"))
    report = on_progress or print
    donors = load_donors()
    df = load_funding_research()
    studied = {normalize_title(name) for name in df["Name"]}
    names = [name for name in unique_donor_names(donors) if normalize_title(name) not in studied]
    if not names:
        report("No unresearched donors remain. Showing saved results.")
        df.attrs["researched_donors"] = 0
        return df

    as_of_date = research_date()
    for number, name in enumerate(names, start=1):
        report(f"Researching {number} of {len(names)}: {name}")
        sources = search_donor_details(name, max_results=max_results, as_of_date=as_of_date)
        report(f"Analyzing funding and application information for {name}...")
        result = analyze_funding(name, sources, as_of_date, on_progress=report)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        df = save_funding_research(result)
        report(f"Research saved for {name}.")
    df.attrs["researched_donors"] = len(names)
    return df


def run_workflow(article_count=1, on_progress=None):
    discovered = run_donor_search(max_results=article_count, on_progress=on_progress)
    researched = run_donor_research(on_progress=on_progress)
    researched.attrs.update(discovered.attrs)
    return researched


def main():
    df = run_workflow()
    print("\nFunding research:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
