import os

from tavily import TavilyClient


def search_event_people(event, organization, max_results=3):
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    company = organization.replace('"', '')
    event_name = event["event_name"][:140]
    queries = [
        f'"{company}" {event_name} founders owners shareholders investors CEO CFO M&A lawyers financial advisers',
        f'"{company}" official leadership family office foundation directors corporate giving philanthropy investor relations corporate development board',
        f'"{company}" executives partners investor relations philanthropy managers professional contact email phone LinkedIn',
    ]
    sources, seen = [], set()
    for query in queries:
        response = client.search(
            query=query[:390], topic="general", search_depth="advanced",
            max_results=max_results, include_raw_content="markdown",
        )
        for source in response.get("results", []):
            url = source.get("url")
            if url and url not in seen:
                seen.add(url)
                sources.append(source)
    return sources
