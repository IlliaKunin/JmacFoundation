import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import donor_search
import main
import storage


def article(number):
    return {"url": f"https://example.org/{number}", "title": f"Source {number}"}


def donor(name):
    return {"name": name, "type": "foundation", "contact_information": {"email": "N/A"}}


def funding(name, applications="Unknown"):
    return {
        "name": name, "funding_match": "Unknown", "applications": applications,
        "contact_method": "N/A", "sources": [],
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.donors = root / "donors.csv"
        self.articles = root / "processed_articles.csv"
        self.research = root / "funding.csv"
        self.details = root / "funding.json"
        self.patch("main.load_dotenv")
        self.patch("main.load_donors", side_effect=lambda: storage.load_donors(self.donors))
        self.patch("main.load_processed_articles", side_effect=lambda: storage.load_processed_articles(self.articles, self.donors))
        self.patch("main.save_processed_article", side_effect=lambda a: storage.save_processed_article(a, self.articles, self.donors))
        self.patch("main.save_donors", side_effect=lambda a, t, r: storage.save_donors(a, t, r, self.donors))
        self.patch("main.load_funding_research", side_effect=lambda: storage.load_funding_research(self.research))
        self.patch("main.save_funding_research", side_effect=lambda r: storage.save_funding_research(r, self.research, self.details))
        self.search = self.patch("main.search_donors", return_value=[])
        self.fetch = self.patch("main.fetch_url", side_effect=lambda url: f"<html>{url}</html>")
        self.patch("main.extract", return_value="Article text")
        self.patch("main.extract_metadata", return_value=None)
        self.extract_donors = self.patch("main.analyze_deal", return_value={"donors": []})
        self.search_details = self.patch("main.search_donor_details", return_value=[])
        self.analyze_funding = self.patch("main.analyze_funding", side_effect=lambda name, sources, date, **kw: funding(name))

    def patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_workflow(self, count):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.run_workflow(article_count=count, on_progress=lambda _: None)

    def test_three_sources_research_all_new_names_and_backlog_without_repeating(self):
        storage.save_donors(article("legacy"), "Legacy text", {"donors": [donor("Known"), donor("Backlog")]}, self.donors)
        storage.save_funding_research(funding(" KNOWN "), self.research, self.details)
        self.search.return_value = [article(1), article(2), article(3), article("extra")]
        extracted = {
            "Source 1": [donor("Alpha"), donor("Beta"), donor("Known")],
            "Source 2": [],
            "Source 3": [donor(" BETA "), donor("Gamma")],
        }
        self.extract_donors.side_effect = lambda text, title, **kw: {"donors": extracted[title]}
        result = self.run_workflow(3)
        self.assertEqual(result.attrs["processed_articles"], 3)
        self.assertEqual(result.attrs["new_donors"], 3)
        self.assertEqual(result.attrs["researched_donors"], 4)
        self.assertEqual(len(result), 5)
        self.assertEqual(self.fetch.call_count, 3)
        self.assertEqual([call.args[0] for call in self.search_details.call_args_list], ["Backlog", "Alpha", "Beta", "Gamma"])
        self.assertEqual(len(pd.read_csv(self.articles)), 4)

        snapshots = {path: path.read_bytes() for path in (self.donors, self.articles, self.research, self.details)}
        calls = self.fetch.call_count, self.extract_donors.call_count, self.analyze_funding.call_count
        self.search.return_value = [article(1), article(2), article(3)]
        repeated = self.run_workflow(3)
        self.assertEqual(repeated.attrs["processed_articles"], 0)
        self.assertEqual(repeated.attrs["researched_donors"], 0)
        self.assertEqual((self.fetch.call_count, self.extract_donors.call_count, self.analyze_funding.call_count), calls)
        for path, contents in snapshots.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_unreadable_pages_are_replaced_and_extra_sources_are_not_processed(self):
        self.search.side_effect = [
            [article("unreadable"), article(1)],
            [article("unreadable"), article(2), article(3), article("extra")],
        ]
        self.fetch.side_effect = lambda url: None if url.endswith("unreadable") else "<html>source</html>"
        result = self.run_workflow(3)
        self.assertEqual(result.attrs["processed_articles"], 3)
        self.assertEqual(result.attrs["search_rounds"], 2)
        self.assertEqual(self.fetch.call_count, 4)
        self.assertEqual(self.extract_donors.call_count, 3)
        self.assertEqual([call.kwargs["max_results"] for call in self.search.call_args_list], [3, 2])
        history = storage.load_processed_articles(self.articles, self.donors)
        self.assertEqual(len(history), 3)
        self.assertFalse(storage.has_article(history, article("unreadable")))

    def test_no_donor_source_is_remembered_and_search_attempts_are_bounded(self):
        self.search.return_value = [article("empty")]
        first = self.run_workflow(1)
        self.assertEqual(first.attrs["processed_articles"], 1)
        self.assertTrue(first.empty)
        second = self.run_workflow(1)
        self.assertEqual(second.attrs["processed_articles"], 0)
        self.assertEqual(second.attrs["search_rounds"], main.MAX_DISCOVERY_ROUNDS)
        self.fetch.assert_called_once()
        self.extract_donors.assert_called_once()
        self.analyze_funding.assert_not_called()

    def test_existing_title_is_skipped_even_with_a_new_url(self):
        storage.save_donors(article("old"), "Text", {"donors": [donor("Known")]}, self.donors)
        storage.save_funding_research(funding("Known"), self.research, self.details)
        self.search.return_value = [{"url": "https://example.org/reprint", "title": " SOURCE   OLD "}]
        result = self.run_workflow(1)
        self.assertEqual(result.attrs["processed_articles"], 0)
        self.fetch.assert_not_called()
        self.extract_donors.assert_not_called()
        self.analyze_funding.assert_not_called()

    def test_duplicate_company_does_not_overwrite_saved_research(self):
        storage.save_funding_research(funding("Example Foundation"), self.research, self.details)
        before_csv, before_json = self.research.read_bytes(), self.details.read_bytes()
        storage.save_funding_research(funding(" EXAMPLE  FOUNDATION ", "Open"), self.research, self.details)
        self.assertEqual(self.research.read_bytes(), before_csv)
        self.assertEqual(self.details.read_bytes(), before_json)


class SearchTests(unittest.TestCase):
    def test_returns_total_source_limit_and_changes_queries_for_refill(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), patch.object(donor_search, "TavilyClient") as client:
            client.return_value.search.return_value = {"results": [article("old"), article(1), article(2), article(3), article(4)]}
            results = donor_search.search_donors(
                ["Query A", "Query B"], max_results=3,
                excluded_urls=[article("old")["url"]], search_round=1,
            )
            self.assertEqual([item["url"] for item in results], [article(i)["url"] for i in (1, 2, 3)])
            client.return_value.search.assert_called_once()
            query = client.return_value.search.call_args.kwargs["query"]
            self.assertTrue(query.startswith("Query B"))
            self.assertNotEqual(query, "Query B")
            self.assertLessEqual(client.return_value.search.call_args.kwargs["max_results"], 20)


class InterfaceTests(unittest.TestCase):
    def test_single_page_uses_source_count_and_shows_all_saved_results(self):
        donors = pd.DataFrame(columns=storage.COLUMNS)
        state = {"funding": pd.DataFrame(columns=storage.FUNDING_COLUMNS)}

        def workflow(article_count, on_progress):
            self.assertEqual(article_count, 3)
            on_progress("Processing three new sources...")
            state["funding"] = pd.DataFrame([
                {"Name": f"Foundation {i}", "Funding Match": "Unknown", "Applications": "Unknown", "Contact Method": ""}
                for i in range(5)
            ])
            state["funding"].attrs.update(processed_articles=3, researched_donors=5, new_donors=5)
            return state["funding"]

        with patch("storage.load_donors", return_value=donors), patch("storage.load_funding_research", side_effect=lambda: state["funding"]), patch("main.run_workflow", side_effect=workflow) as run:
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ["Donor Research", "News Search"])
            self.assertEqual(len(app.number_input), 2)
            self.assertEqual(len(app.button), 3)
            self.assertFalse(app.get("download_button"))
            self.assertFalse(app.get("json"))
            run.assert_not_called()
            app.number_input(key="article_count").set_value(3).run(timeout=20)
            run.assert_not_called()
            app.button(key="start_workflow").click().run(timeout=20)
            self.assertFalse(app.exception)
            run.assert_called_once()
            self.assertEqual(len(app.dataframe[0].value), 5)
            self.assertEqual(len(app.success), 1)
            app.run(timeout=20)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(app.dataframe[0].value), 5)


if __name__ == "__main__":
    unittest.main()
