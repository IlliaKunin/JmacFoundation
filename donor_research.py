import os
from datetime import datetime
from zoneinfo import ZoneInfo

from tavily import TavilyClient


def research_date():
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def search_donor_details(name, max_results=3, as_of_date=None):
    if not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be an integer between 1 and 20.")
    name = name.strip()
    if not name:
        raise ValueError("A donor name is required.")
    as_of_date = as_of_date or research_date()
    queries = [
        f'"{name}" official funding priorities education AI STEM youth grant programs',
        f'"{name}" official grants apply applications open invitation only '
        f'grant amounts deadlines eligibility geographic restrictions contact {as_of_date}',
    ]
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    sources = []
    seen_urls = set()
    for query in queries:
        response = client.search(
            query=query,
            topic="general",
            search_depth="advanced",
            max_results=max_results,
            include_raw_content="markdown",
        )
        for source in response.get("results", []):
            url = source.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(source)
    return sources
