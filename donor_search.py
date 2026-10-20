import json
import os
from pathlib import Path
from tavily import TavilyClient
from storage import normalize_title


DONOR_QUERY = [
    # Foundations that give grants
    "US private family foundations awarding grants to education youth STEM nonprofits 2025 2026",

    # Individuals who have donated money
    "US wealthy individual philanthropists major charitable donations education children youth 2025 2026",

    # Actual donor lists and reports
    "US nonprofit donor honor rolls major donors family charitable funds annual reports 2024 2025 2026"
]



SEARCH_FOCUS = [
    "",
    "official donor lists and philanthropic grant programs",
    "foundation annual reports and named charitable contributors",
    "major individual donors education youth STEM charitable gifts",
    "private family foundations nonprofit grants donor honor rolls",
]


def search_donors(query=DONOR_QUERY, max_results=3, excluded_titles=(), excluded_urls=(), search_round=0):
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be an integer between 1 and 20.")
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    queries = [query] if isinstance(query, str) else list(query)
    if queries:
        offset = search_round % len(queries)
        queries = queries[offset:] + queries[:offset]
    results = []
    seen_urls = set(excluded_urls)
    seen_titles = {normalize_title(title) for title in excluded_titles}
    seen_titles.discard("")
    focus = SEARCH_FOCUS[search_round % len(SEARCH_FOCUS)]
    candidate_limit = min(20, max_results + min(len(seen_urls), 10) + search_round * 2)
    for search_query in queries:
        if focus:
            search_query = f"{search_query} {focus}"
        response = client.search(
            query=search_query,
            topic="general",
            search_depth="advanced",
            max_results=candidate_limit,
        )
        for article in response.get("results", []):
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
    from storage import load_processed_articles

    load_dotenv(Path(__file__).with_name(".env"))
    saved = load_processed_articles()
    donors = search_donors(
        excluded_titles=saved["article_title"],
        excluded_urls=saved["url"],
    )

    output_path = Path(__file__).with_name("donor_results.json")
    output_path.write_text(
        json.dumps({"query": DONOR_QUERY, "results": donors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Query:", DONOR_QUERY)
    print(f"Found: {len(donors)} results")
    for number, donor in enumerate(donors, start=1):
        print(f"\n--- {number} ---")
        print("Title:", donor.get("title"))
        print("URL:", donor.get("url"))
        print("Score:", donor.get("score"))
        print("Content:", donor.get("content", ""))

    print(f"\nSaved search results to {output_path}")
