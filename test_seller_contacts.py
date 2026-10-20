import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import seller_contacts as contacts
import storage
from deal_analyzer import MAX_INPUT_TOKENS, _message_tokens


def participant(name="Jane Doe", kind="individual", organization="Acme", side="Seller", url="https://news.example.org/deal"):
    return {
        "participant_name": name, "participant_type": kind, "organization": organization,
        "role": "CEO of Acme", "deal_side": side, "article_title": "Acme sale",
        "description": "Acme is selling a business.", "url": url,
    }


BIO_URL = "https://acme.example.org/team/jane"
LINKEDIN = "https://www.linkedin.com/in/jane-doe"
CONTACT_PAGE = "https://acme.example.org/team/jane/contact"
BIO = (
    f"Jane Doe is the CEO of Acme. Contact Jane Doe at jane@acme.example.org or "
    f"+1 (212) 555-0100. Contact page: {CONTACT_PAGE}. LinkedIn: {LINKEDIN}."
)


def response(**overrides):
    return {
        "identity_match": True,
        "email": "jane@acme.example.org", "email_source_url": BIO_URL, "email_scope": "individual",
        "phone": "+1 (212) 555-0100", "phone_source_url": BIO_URL, "phone_scope": "individual",
        "contact_page": CONTACT_PAGE, "contact_page_source_url": BIO_URL,
        "linkedin": LINKEDIN, "linkedin_source_url": BIO_URL,
        **overrides,
    }


class ContactAnalysisTests(unittest.TestCase):
    def analyze(self, payload, row=None, sources=None):
        with patch.object(contacts, "_request_json", return_value=payload):
            return contacts.analyze_seller_contacts(
                row or participant(), sources or [{"url": BIO_URL, "title": "Jane Doe at Acme", "content": BIO}],
            )

    def test_published_contacts_and_actual_linkedin_are_extracted(self):
        result = self.analyze(response())
        self.assertEqual(result["email"], "jane@acme.example.org")
        self.assertEqual(result["phone"], "+1 (212) 555-0100")
        self.assertEqual(result["contact_page"], CONTACT_PAGE)
        self.assertEqual(result["linkedin"], LINKEDIN)
        self.assertEqual(result["contact_sources"], BIO_URL)

    def test_guessed_contacts_and_company_contacts_for_person_are_rejected(self):
        result = self.analyze(response(
            email="guessed@acme.example.org", phone_scope="organization",
            contact_page="https://acme.example.org/guessed", linkedin="https://www.linkedin.com/company/acme",
        ))
        self.assertTrue(all(value == "" for value in result.values()))

    def test_wrong_identity_and_unmatched_source_leave_fields_blank(self):
        self.assertTrue(all(value == "" for value in self.analyze(response(identity_match=False)).values()))
        result = self.analyze(response(), sources=[{"url": BIO_URL, "title": "Other Person", "content": "Contact someone else at jane@acme.example.org"}])
        self.assertTrue(all(value == "" for value in result.values()))

    def test_organization_uses_organization_contacts_and_company_linkedin(self):
        profile = "https://www.linkedin.com/company/acme"
        result = self.analyze(response(
            email_scope="organization", phone_scope="organization", linkedin=profile,
        ), row=participant("Acme", "company"), sources=[{"url": BIO_URL, "title": "Acme contacts", "content": BIO + " " + profile}])
        self.assertEqual(result["email"], "jane@acme.example.org")
        self.assertEqual(result["linkedin"], profile)

    def test_invalid_linkedin_hosts_and_missing_values_are_blank(self):
        for profile in ("https://linkedin.com.fake.example/in/jane", "https://www.linkedin.com/search/results/people/", "N/A"):
            with self.subTest(profile=profile):
                self.assertEqual(self.analyze(response(linkedin=profile))["linkedin"], "")

    def test_large_sources_stay_inside_groq_input_budget(self):
        sources = [{"url": f"https://example.org/{i}", "title": "Source", "raw_content": 'Long "source" text.\n' * 10000} for i in range(9)]
        with patch.object(contacts, "_request_json", return_value={"identity_match": False}) as request:
            result = contacts.analyze_seller_contacts(participant(), sources)
        self.assertTrue(all(value == "" for value in result.values()))
        self.assertLessEqual(_message_tokens(request.call_args.args[0]), MAX_INPUT_TOKENS)

    def test_no_sources_does_not_call_groq(self):
        with patch.object(contacts, "_request_json") as request:
            result = contacts.analyze_seller_contacts(participant(), [])
        request.assert_not_called()
        self.assertEqual(result["linkedin"], "")


class SellerContactWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "contacts.csv"
        self.news = pd.DataFrame([
            participant(), participant(url="https://news.example.org/another"),
            participant("Buyer", "company", "Buyer", "Buyer"),
            participant("Both sides", "company", "Both sides", "Both"),
            participant("Adviser", "company", "Adviser"),
        ])
        self.patch("load_news_articles", side_effect=lambda: self.news)
        self.patch("load_seller_contacts", side_effect=lambda: storage.load_seller_contacts(self.path))
        self.patch("save_seller_contact", side_effect=lambda row, data, date: storage.save_seller_contact(row, data, date, self.path))
        self.search = self.patch("search_seller_contacts", return_value=[])
        self.analyze = self.patch("analyze_seller_contacts", return_value={field: "" for field in contacts.CONTACT_FIELDS})

    def patch(self, name, **kwargs):
        patcher = patch.object(contacts, name, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def test_only_seller_identities_are_checked_once_including_empty_results(self):
        result = contacts.run_seller_contacts(on_progress=lambda _: None)
        self.assertEqual(result.attrs["checked_sellers"], 2)
        self.assertEqual(set(result["participant_name"]), {"Jane Doe", "Adviser"})
        self.assertEqual(self.search.call_count, 2)
        repeated = contacts.run_seller_contacts(on_progress=lambda _: None)
        self.assertEqual(repeated.attrs["checked_sellers"], 0)
        self.assertEqual(self.search.call_count, 2)
        self.assertEqual(len(storage.load_seller_contacts(self.path)), 2)

    def test_name_and_organization_separate_namesakes(self):
        self.news = pd.DataFrame([participant(), participant(organization="Other company")])
        self.assertEqual(len(contacts.seller_entities(self.news)), 2)

    def test_partial_results_survive_failure_and_retry_skips_completed_identity(self):
        self.analyze.side_effect = [{field: "" for field in contacts.CONTACT_FIELDS}, RuntimeError("Simulated failure")]
        with self.assertRaises(RuntimeError):
            contacts.run_seller_contacts(on_progress=lambda _: None)
        self.assertEqual(len(storage.load_seller_contacts(self.path)), 1)
        self.analyze.side_effect = None
        self.analyze.return_value = {field: "" for field in contacts.CONTACT_FIELDS}
        self.assertEqual(contacts.run_seller_contacts(on_progress=lambda _: None).attrs["checked_sellers"], 1)
        self.assertEqual(self.search.call_count, 3)


class ContactSearchTests(unittest.TestCase):
    def test_queries_use_identity_and_linkedin_domain_without_news_time_filter(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), patch.object(contacts, "TavilyClient") as client:
            client.return_value.search.return_value = {"results": [{"url": BIO_URL, "content": BIO}]}
            sources = contacts.search_seller_contacts(participant())
        self.assertEqual(len(sources), 1)
        self.assertEqual(client.return_value.search.call_count, 3)
        calls = client.return_value.search.call_args_list
        self.assertTrue(all('"Jane Doe"' in call.kwargs["query"] and '"Acme"' in call.kwargs["query"] for call in calls))
        self.assertEqual(calls[-1].kwargs["include_domains"], ["linkedin.com"])
        self.assertTrue(all(call.kwargs["topic"] == "general" and "days" not in call.kwargs for call in calls))


class SellerContactsInterfaceTests(unittest.TestCase):
    def test_single_search_and_table_include_article_context_and_linkedin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contacts.csv"
            news = pd.DataFrame([participant(), participant("Buyer", "company", "Buyer", "Buyer")])
            load_cached = storage.load_seller_contacts

            def run(article_count, on_progress):
                self.assertEqual(article_count, 1)
                on_progress("Finding seller contacts...")
                storage.save_seller_contact(news.iloc[0].to_dict(), {
                    "email": "", "phone": "", "contact_page": "", "linkedin": LINKEDIN, "contact_sources": BIO_URL,
                }, "2026-10-20", path)
                result = news.copy()
                result.attrs.update(checked_sellers=1, saved_articles=1, processed_articles=1)
                return result

            with patch("storage.load_news_articles", return_value=news), patch("storage.load_seller_contacts", side_effect=lambda *args: load_cached(path)), patch("seller_contacts.run_seller_contacts") as separate_search, patch("main.run_workflow") as donors, patch("news_workflow.run_news_workflow", side_effect=run) as search_news:
                app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(len(app.tabs[1].dataframe), 1)
                self.assertEqual(len(app.tabs[1].button), 2)  # Search and delete.
                search_news.assert_not_called()
                self.assertNotIn("proceeds_recipient", app.tabs[1].dataframe[0].value.columns)
                self.assertEqual(list(app.tabs[1].dataframe[0].value["participant_name"]), ["Jane Doe"])
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[0]["linkedin"], "")
                app.button(key="start_news_search").click().run(timeout=20)
                self.assertFalse(app.exception)
                search_news.assert_called_once()
                self.assertEqual(len(app.tabs[1].dataframe), 1)
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[0]["linkedin"], LINKEDIN)
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[0]["article_title"], "Acme sale")
                app.run(timeout=20)
                search_news.assert_called_once()
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[0]["linkedin"], LINKEDIN)
                donors.assert_not_called()
                separate_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
