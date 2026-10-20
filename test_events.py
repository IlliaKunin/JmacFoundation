import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import event_analyzer
import event_workflow
import news_search
import storage
from deal_analyzer import MAX_INPUT_TOKENS, _message_tokens


ARTICLE = {"url": "https://news.example.org/deal", "title": "Acme buys Beta", "published_date": "2026-10-19"}
TEXT = "Acme is a US-based company. Acme acquired Beta. Acme CEO Jane Doe led the acquisition."
EVENT = {
    "event_name": "Acme acquisition of Beta", "event_type": "M&A",
    "description": "Acme acquired Beta.", "source_excerpt": TEXT,
    "participants": [
        {"name": "Acme", "entity_type": "organization", "role": "buyer", "company": ""},
        {"name": "Beta", "entity_type": "organization", "role": "target", "company": ""},
    ],
}
BIO_URL = "https://acme.example.org/team/jane"
PROFILE = "https://www.linkedin.com/in/jane-doe"
BIO = (
    "Jane Doe is the CEO of Acme. "
    "Jane Doe's work email is jane@acme.example.org. "
    "Jane Doe's office phone is +1 (212) 555-0100. "
    f"Jane Doe's LinkedIn profile is {PROFILE}."
)


def event_record():
    return {**EVENT, "event_id": "test-event", "article_url": ARTICLE["url"], "article_title": ARTICLE["title"]}


def person_payload():
    return {
        "name": "Jane Doe", "company": "Acme", "position": "CEO",
        "connection_type": "organization", "related_organization": "Acme",
        "event_connection": "CEO of the buyer Acme.",
        "relationship_evidence": "Jane Doe is the CEO of Acme.",
        "email": "jane@acme.example.org", "email_source_url": BIO_URL,
        "email_evidence": "Jane Doe's work email is jane@acme.example.org.",
        "phone": "+1 (212) 555-0100", "phone_source_url": BIO_URL,
        "phone_evidence": "Jane Doe's office phone is +1 (212) 555-0100.",
        "linkedin": PROFILE, "linkedin_source_url": BIO_URL, "source_urls": [BIO_URL],
    }


class EventAnalysisTests(unittest.TestCase):
    def test_event_requires_us_evidence_and_source_names(self):
        payload = {**EVENT, "us_company_evidence": "Acme is a US-based company."}
        payload["participants"] = [*EVENT["participants"], {"name": "Invented Person", "entity_type": "person"}]
        with patch.object(event_analyzer, "_request_json", return_value={"events": [payload, {**payload, "us_company_evidence": "Not in the text"}]}) as request:
            events = event_analyzer.analyze_events(ARTICLE, TEXT)
        self.assertEqual(len(events), 1)
        self.assertEqual([actor["name"] for actor in events[0]["participants"]], ["Acme", "Beta"])
        self.assertIn("Acme", events[0]["source_excerpt"])
        self.assertLessEqual(_message_tokens(request.call_args.args[0]), MAX_INPUT_TOKENS)

    def test_supported_professional_contacts_and_affiliation_explanation(self):
        sources = [{"url": BIO_URL, "title": "Jane Doe - Acme", "content": BIO}]
        with patch.object(event_analyzer, "_request_json", return_value={"people": [person_payload()]}) as request:
            people = event_analyzer.analyze_event_people(event_record(), "Acme", sources)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["email"], "jane@acme.example.org")
        self.assertEqual(people[0]["phone"], "+1 (212) 555-0100")
        self.assertEqual(people[0]["linkedin"], PROFILE)
        self.assertIn("direct transaction participation not confirmed", people[0]["event_connection"])
        self.assertIn(BIO_URL, people[0]["source_urls"])
        self.assertLessEqual(_message_tokens(request.call_args.args[0]), MAX_INPUT_TOKENS)

    def test_guessed_contacts_company_profiles_and_unrelated_people_are_rejected(self):
        invalid = {
            **person_payload(), "email": "guessed@acme.example.org",
            "phone_evidence": "Acme's general phone is +1 (212) 555-0100.",
            "linkedin": "https://www.linkedin.com/company/acme",
        }
        unrelated = {**person_payload(), "related_organization": "Unrelated Foundation"}
        invented = {**person_payload(), "name": "Invented Person"}
        sources = [{"url": BIO_URL, "title": "Acme leadership", "content": BIO}]
        with patch.object(event_analyzer, "_request_json", return_value={"people": [invalid, unrelated, invented]}):
            people = event_analyzer.analyze_event_people(event_record(), "Acme", sources)
        self.assertEqual(len(people), 1)
        self.assertEqual([people[0][key] for key in ("email", "phone", "linkedin")], ["", "", ""])

    def test_many_large_sources_stay_inside_groq_token_budget(self):
        sources = [{"url": f"https://example.org/{i}", "content": ('Quote "text" \n' * 10000)} for i in range(12)]
        with patch.object(event_analyzer, "_request_json", return_value={"people": []}) as request:
            self.assertEqual(event_analyzer.analyze_event_people(event_record(), "Acme", sources), [])
        self.assertLessEqual(_message_tokens(request.call_args.args[0]), MAX_INPUT_TOKENS)
        json.loads(request.call_args.args[0][1]["content"])


class EventStorageTests(unittest.TestCase):
    def test_events_people_and_missing_contacts_survive_repeated_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.csv"
            people_path = Path(directory) / "people.csv"
            events = storage.save_events(ARTICLE, [EVENT], events_path)
            self.assertEqual(len(storage.save_events(ARTICLE, [EVENT], events_path)), 1)
            event = {**events.iloc[0].to_dict(), "participants": EVENT["participants"]}
            person = {"name": "Jane Doe", "company": "Acme", "position": "CEO", "event_connection": "CEO of participant Acme", "source_urls": [ARTICLE["url"]]}
            people = storage.save_event_people(event, [person], people_path)
            self.assertEqual(people.iloc[0]["email"], "")
            storage.save_event_people(event, [{**person, "email": "jane@acme.example.org", "source_urls": [BIO_URL]}], people_path)
            reloaded = storage.load_event_people(people_path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded.iloc[0]["email"], "jane@acme.example.org")
            self.assertEqual(reloaded.iloc[0]["linkedin"], "")
            self.assertIn(BIO_URL, reloaded.iloc[0]["source_urls"])
            storage.mark_event_researched(event["event_id"], events_path)
            self.assertEqual(storage.load_events(events_path).iloc[0]["research_complete"], "yes")


class NewsSearchTests(unittest.TestCase):
    def test_news_categories_are_interleaved_with_month_filter(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), patch.object(news_search, "NEWS_QUERIES", ["M&A query", "Funding query", "IPO query"]), patch.object(news_search, "TavilyClient") as client:
            client.return_value.search.side_effect = [
                {"results": [{"url": "https://example.org/old", "title": "Saved"}, {"url": "https://example.org/ma", "title": "M&A"}]},
                {"results": [{"url": "https://example.org/round", "title": "Funding round"}]},
                {"results": [{"url": "https://example.org/ipo", "title": "IPO"}]},
            ]
            found = news_search.search_news(max_results=3, excluded_titles=[" SAVED "])
            self.assertEqual(len(found), 3)
            self.assertEqual({item["title"] for item in found}, {"M&A", "Funding round", "IPO"})
            self.assertEqual(client.return_value.search.call_count, 3)
            for call in client.return_value.search.call_args_list:
                self.assertEqual(call.kwargs["days"], 30)
                self.assertEqual(call.kwargs["topic"], "news")


class EventWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.events = root / "events.csv"
        self.people = root / "people.csv"
        self.history = root / "history.csv"
        self.patch("load_dotenv")
        self.patch("load_events", side_effect=lambda: storage.load_events(self.events))
        self.patch("load_event_people", side_effect=lambda: storage.load_event_people(self.people))
        self.patch("load_event_articles", side_effect=lambda: storage.load_event_articles(self.history))
        self.patch("save_events", side_effect=lambda a, e: storage.save_events(a, e, self.events))
        self.patch("save_event_people", side_effect=lambda e, p: storage.save_event_people(e, p, self.people))
        self.patch("save_event_article", side_effect=lambda a: storage.save_event_article(a, self.history))
        self.patch("mark_event_researched", side_effect=lambda event_id: storage.mark_event_researched(event_id, self.events))
        self.news = self.patch("search_news", return_value=[{**ARTICLE, "raw_content": TEXT}])
        self.extract_events = self.patch("analyze_events", return_value=[EVENT])
        self.research = self.patch("search_event_people", return_value=[])
        self.analyze_people = self.patch("analyze_event_people", return_value=[{
            "name": "Jane Doe", "company": "Acme", "position": "CEO", "event_connection": "Affiliated with Acme", "source_urls": [ARTICLE["url"]],
        }])

    def patch(self, name, **kwargs):
        patcher = patch.object(event_workflow, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_full_pipeline_and_repeat_skip(self):
        result = event_workflow.run_event_workflow(on_progress=lambda _: None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.attrs["new_events"], 1)
        self.assertEqual(result.attrs["new_people"], 1)
        self.assertEqual(self.research.call_count, 2)
        self.assertEqual(self.news.call_args.kwargs["days"], 14)
        repeated = event_workflow.run_event_workflow(on_progress=lambda _: None)
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated.attrs["new_events"], 0)
        self.assertEqual(repeated.attrs["researched_events"], 0)
        self.extract_events.assert_called_once()
        self.assertEqual(self.research.call_count, 2)

    def test_no_qualifying_event_is_still_remembered(self):
        self.extract_events.return_value = []
        self.assertTrue(event_workflow.run_event_workflow(on_progress=lambda _: None).empty)
        event_workflow.run_event_workflow(on_progress=lambda _: None)
        self.extract_events.assert_called_once()
        self.research.assert_not_called()

    def test_failed_people_research_can_resume_saved_event(self):
        self.research.side_effect = RuntimeError("Simulated research failure")
        with self.assertRaises(RuntimeError):
            event_workflow.run_event_workflow(on_progress=lambda _: None)
        self.assertEqual(storage.load_events(self.events).iloc[0]["research_complete"], "no")
        self.research.side_effect = None
        self.research.return_value = []
        resumed = event_workflow.run_event_workflow(on_progress=lambda _: None)
        self.assertEqual(len(resumed), 1)
        self.extract_events.assert_called_once()
        self.assertEqual(storage.load_events(self.events).iloc[0]["research_complete"], "yes")


if __name__ == "__main__":
    unittest.main()
