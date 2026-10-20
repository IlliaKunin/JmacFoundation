import json
import os
from pathlib import Path

from tavily import TavilyClient
from storage import normalize_title


NEWS_SEARCH_DAYS = 30

NEWS_QUERIES = [
    "United States founder owner sells business acquisition deal announced purchase price named seller",
    "United States company announces sale of subsidiary assets cash proceeds named seller",
    "United States founders sell existing shares secondary sale cash out transaction announcement",
    "United States private equity firm exits portfolio company investment sale transaction announced",
    "United States family owned business sold acquisition founders owners named sellers purchase price",
    "United States cash acquisition shareholders receive cash consideration merger agreement announced closed",
]


def search_news(query=None, max_results=1, days=NEWS_SEARCH_DAYS, excluded_titles=(), excluded_urls=(), search_round=0):
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be an integer between 1 and 20.")
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer.")
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    queries = NEWS_QUERIES if query is None else ([query] if isinstance(query, str) else list(query))
    if queries:
        offset = search_round % len(queries)
        queries = queries[offset:] + queries[:offset]
    seen_urls = set(excluded_urls)
    seen_titles = {normalize_title(title) for title in excluded_titles} - {""}
    results = []
    batches = []
    for search_query in queries:
        if search_round:
            search_query += " company press release transaction announcement"
        response = client.search(
            query=search_query, topic="news", days=days,
            search_depth="advanced", include_raw_content="markdown",
            max_results=min(20, max_results + min(len(seen_urls), 10) + search_round * 2),
        )
        batches.append(response.get("results", []))
    # Interleave the queries rather than taking every result from the first one.
    for rank in range(max((len(batch) for batch in batches), default=0)):
        for batch in batches:
            if rank >= len(batch):
                continue
            article = batch[rank]
            url = article.get("url")
            title = normalize_title(article.get("title"))
            if not url or url in seen_urls or (title and title in seen_titles):
                continue
            seen_urls.add(url)
            if title:
                seen_titles.add(title)
            results.append(article)
            if len(results) >= max_results:
                return results
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
    articles = search_news()

    output_path = Path(__file__).with_name("news_results.json")
    output_path.write_text(
        json.dumps({"queries": NEWS_QUERIES, "results": articles}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Queries:", NEWS_QUERIES)
    print(f"Found: {len(articles)} results")
    for number, article in enumerate(articles, start=1):
        print(f"\n--- {number} ---")
        print("Title:", article.get("title"))
        print("URL:", article.get("url"))
        print("Publication date:", article.get("published_date"))
        print("Score:", article.get("score"))
        print("Content:", article.get("content", ""))

    print(f"\nSaved search results to {output_path}")
