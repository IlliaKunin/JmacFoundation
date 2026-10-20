import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import deal_analyzer
import news_workflow
import seller_contacts
import storage


def article(number, **overrides):
    return {
        "title": f"Transaction {number}", "url": f"https://example.org/news/{number}",
        "published_date": "Wed, 07 Oct 2026 20:30:00 GMT",
        "raw_content": f"Company {number}, based in Texas, sold its subsidiary and received $20 million in cash.",
        **overrides,
    }


def decision(number=1, **overrides):
    return {
        "qualifies": True,
        "us_scope": "Yes",
        "us_reason": f"The source identifies Company {number} as based in Texas.",
        "description": "Groq description.",
        "reason": "A named seller received sale proceeds.",
        "participants": [{
            "name": f"Company {number}", "type": "company", "deal_side": "Seller",
        }],
        **overrides,
    }


class SummaryTests(unittest.TestCase):
    def test_long_text_is_cut_at_token_budget_and_summary_comes_from_groq(self):
        text = article(1)["raw_content"] + " Beginning of article. " + 'Large article with "quotes".\n' * 12000
        with patch.object(deal_analyzer, "_request_json", return_value=decision()) as request:
            result = deal_analyzer.analyze_news(text, "Subsidiary sale")
            self.assertEqual(result["description"], "Groq description.")
            self.assertTrue(result["qualifies"])
            self.assertEqual(result["participants"][0]["name"], "Company 1")
        messages = request.call_args.args[0]
        self.assertLessEqual(deal_analyzer._message_tokens(messages), deal_analyzer.MAX_INPUT_TOKENS)
        self.assertIn("Beginning of article.", messages[1]["content"])
        self.assertLess(len(messages[1]["content"]), len(text))

    def test_invalid_summary_is_not_accepted(self):
        for payload in ({}, decision(description=" "), decision(description=[]), decision(qualifies="false"), decision(reason=""), decision(participants="Company 1"), []):
            with self.subTest(payload=payload), patch.object(deal_analyzer, "_request_json", return_value=payload):
                with self.assertRaises(ValueError):
                    deal_analyzer.analyze_news("Article text")

    def test_relevant_sale_without_recipient_or_payout_proof_can_be_saved(self):
        with patch.object(deal_analyzer, "_request_json", return_value=decision(participants=[])):
            result = deal_analyzer.analyze_news("Texas-based Acme announced its acquisition of Beta.")
        self.assertTrue(result["qualifies"])
        self.assertFalse(result["participants"])

    def test_explicit_model_exclusion_is_respected(self):
        payload = decision(qualifies=False, reason="This is a business loan.", participants=[])
        with patch.object(deal_analyzer, "_request_json", return_value=payload):
            self.assertFalse(deal_analyzer.analyze_news("Company 1 obtained a bank loan.")["qualifies"])

    def test_announced_cash_sale_with_expected_recipient_is_eligible(self):
        text = "Jane Doe agreed to sell her shares in Texas-based Acme and will receive $15 million in cash upon closing."
        payload = decision(
            participants=[{"name": "Jane Doe", "type": "individual"}],
        )
        with patch.object(deal_analyzer, "_request_json", return_value=payload):
            result = deal_analyzer.analyze_news(text)
        self.assertTrue(result["qualifies"])
        self.assertEqual(result["participants"][0]["name"], "Jane Doe")

    def test_seller_from_title_is_used_when_page_is_blocked(self):
        title = "Kevin O'Leary sold his US-based software company and walked away with about $11 million"
        payload = decision(participants=[{"name": "Kevin O'Leary", "type": "individual"}])
        with patch.object(deal_analyzer, "_request_json", return_value=payload):
            result = deal_analyzer.analyze_news("Please accept cookies to continue.", title)
        self.assertTrue(result["qualifies"])
        self.assertEqual(result["participants"][0]["name"], "Kevin O'Leary")

    def test_unseen_names_are_removed_without_rejecting_the_sale(self):
        payload = decision(participants=[{"name": "Invented Person", "type": "individual"}])
        with patch.object(deal_analyzer, "_request_json", return_value=payload):
            result = deal_analyzer.analyze_news(article(1)["raw_content"])
        self.assertTrue(result["qualifies"])
        self.assertFalse(result["participants"])

    def test_names_in_article_tail_are_extracted_in_additional_chunks(self):
        text = ("Unrelated source information. " * 12000) + article(1)["raw_content"]
        with patch.object(deal_analyzer, "_request_json", return_value=decision()):
            self.assertEqual(deal_analyzer.analyze_news(text)["participants"][0]["name"], "Company 1")

    def test_foreign_or_unknown_geography_excludes_otherwise_relevant_sale(self):
        for scope in ("No", "Unknown", None, "invalid"):
            payload = decision(us_scope=scope, us_reason="The source does not establish a US business connection.")
            with self.subTest(scope=scope), patch.object(deal_analyzer, "_request_json", return_value=payload):
                result = deal_analyzer.analyze_news("Company 1 sold a business based in London.")
            self.assertFalse(result["qualifies"])
            self.assertFalse(result["participants"])
            self.assertIn("US business filter", result["reason"])

    def test_unexplained_us_flag_is_not_sufficient(self):
        with patch.object(deal_analyzer, "_request_json", return_value=decision(us_reason="")):
            result = deal_analyzer.analyze_news("Company 1 announced an acquisition.")
        self.assertFalse(result["qualifies"])
        self.assertEqual(result["us_scope"], "Unknown")

    def test_cross_border_deals_with_us_buyer_or_us_target_can_qualify(self):
        for text, reason in (
            ("Texas-based Company 1 acquired a German business.", "Company 1, the buyer, is based in Texas."),
            ("A British company acquired Company 1, a Florida-based business.", "Company 1, the target, is based in Florida."),
        ):
            with self.subTest(text=text), patch.object(deal_analyzer, "_request_json", return_value=decision(us_reason=reason)):
                result = deal_analyzer.analyze_news(text)
            self.assertTrue(result["qualifies"])
            self.assertEqual(result["us_scope"], "Yes")

    def test_people_companies_foundations_and_funds_are_extracted_separately(self):
        text = (
            "Texas-based Acme announced an acquisition. Jane Doe, Seller Holdings, "
            "Smith Foundation and Growth Fund will receive cash for their shares."
        )
        participants = [
            {"name": "Jane Doe", "type": "individual"},
            {"name": "Seller Holdings", "type": "company"},
            {"name": "Smith Foundation", "type": "foundation"},
            {"name": "Growth Fund", "type": "fund"},
        ]
        with patch.object(deal_analyzer, "_request_json", return_value=decision(participants=[
            *participants, {"name": " JANE DOE ", "type": "individual"},
            {"name": "Invented Fund", "type": "fund"},
        ])):
            result = deal_analyzer.analyze_news(text)
        self.assertEqual(result["participants"], [{**item, "organization": item['name'] if item['type'] != 'individual' else '', "role": "", "deal_side": "Unknown", "proceeds_recipient": "Unknown"} for item in participants])

    def test_seller_family_chair_and_advisers_are_all_retained_with_distinct_roles(self):
        text = (
            "Wisconsin-based Brakebush is being sold by the Brakebush family. "
            "Carey Brakebush is its chairman. William Blair advises the seller financially. "
            "Michael Best & Friedrich LLP is the seller's legal counsel. "
            "Wells Fargo advises the buyer financially."
        )
        entries = [
            {"name": "Wells Fargo", "type": "company", "role": "Buyer's financial adviser", "deal_side": "Buyer", "proceeds_recipient": "No"},
            {"name": "Carey Brakebush", "type": "individual", "role": "Chairman of the selling company", "deal_side": "Seller", "proceeds_recipient": "Unknown"},
            {"name": "Brakebush family", "type": "family", "role": "Owning family selling the business", "deal_side": "Seller", "proceeds_recipient": "Yes"},
            {"name": "William Blair", "type": "company", "role": "Seller's financial adviser", "deal_side": "Seller", "proceeds_recipient": "No"},
            {"name": "Michael Best & Friedrich LLP", "type": "company", "role": "Seller's legal counsel", "deal_side": "Seller", "proceeds_recipient": "No"},
        ]
        with patch.object(deal_analyzer, "_request_json", return_value=decision(participants=entries)):
            result = deal_analyzer.analyze_news(text)
        self.assertEqual(len(result["participants"]), 5)
        self.assertEqual(result["participants"][-1]["deal_side"], "Buyer")
        by_name = {entry["name"]: entry for entry in result["participants"]}
        self.assertEqual(by_name["William Blair"]["role"], "Seller's financial adviser")
        self.assertEqual(by_name["William Blair"]["proceeds_recipient"], "No")
        self.assertEqual(by_name["Carey Brakebush"]["proceeds_recipient"], "Unknown")
        self.assertEqual(by_name["Brakebush family"]["type"], "family")

    def test_long_article_advisers_at_end_are_extracted_within_request_limits(self):
        text = article(1)["raw_content"] + " Additional business details. " * 3000 + " William Blair advises the seller."
        adviser = {
            "name": "William Blair", "type": "company", "role": "Seller's financial adviser",
            "deal_side": "Seller", "proceeds_recipient": "No",
        }
        def response(messages, **kwargs):
            if "Article continuation:" in messages[1]["content"]:
                return {"participants": [adviser]}
            return decision()
        with patch.object(deal_analyzer, "_request_json", side_effect=response) as request:
            result = deal_analyzer.analyze_news(text)
        self.assertGreater(request.call_count, 1)
        self.assertTrue(all(deal_analyzer._message_tokens(call.args[0]) <= deal_analyzer.MAX_INPUT_TOKENS for call in request.call_args_list))
        by_name = {entry["name"]: entry for entry in result["participants"]}
        self.assertIn("Company 1", by_name)
        self.assertEqual(by_name["William Blair"]["deal_side"], "Seller")


class NewsWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "news.csv"
        self.history_path = Path(temporary.name) / "history.csv"
        self.patch("load_news_articles", side_effect=lambda: storage.load_news_articles(self.path))
        self.patch("save_news_article", side_effect=lambda a, d, **kwargs: storage.save_news_article(a, d, self.path, **kwargs))
        self.patch("load_news_history", side_effect=lambda: storage.load_news_history(self.history_path))
        self.patch("save_news_decision", side_effect=lambda a, d: storage.save_news_decision(a, d, self.history_path))
        self.search = self.patch("search_news", return_value=[article(1), article(2)])
        self.analyze = self.patch("analyze_news", return_value=decision())
        self.fetch = self.patch("fetch_url", return_value=None)
        self.contacts_path = Path(temporary.name) / "contacts.csv"
        for target, options in (
            ("load_seller_contacts", {"side_effect": lambda: storage.load_seller_contacts(self.contacts_path)}),
            ("save_seller_contact", {"side_effect": lambda row, data, date: storage.save_seller_contact(row, data, date, self.contacts_path)}),
            ("search_seller_contacts", {"return_value": []}),
            ("analyze_seller_contacts", {"return_value": {field: "" for field in seller_contacts.CONTACT_FIELDS}}),
        ):
            patcher = patch.object(seller_contacts, target, **options)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            setattr(self, target, mock)

    def patch(self, name, **kwargs):
        patcher = patch.object(news_workflow, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_workflow(self, count):
        return news_workflow.run_news_workflow(count, on_progress=lambda _: None)

    def test_new_rows_persist_and_repeated_urls_or_titles_skip_groq(self):
        self.analyze.side_effect = [decision(1), decision(2)]
        first = self.run_workflow(2)
        self.assertEqual(len(first), 2)
        self.assertEqual(first.attrs["processed_articles"], 2)
        self.assertEqual(set(first["published_date"]), {"07 Oct 2026"})
        self.assertEqual(set(first["description"]), {"Groq description."})
        self.assertEqual(set(first["participant_name"]), {"Company 1", "Company 2"})
        self.assertEqual(first.attrs["stop_reason"], "target_reached")
        self.assertTrue(all(call.kwargs["days"] == 30 for call in self.search.call_args_list))
        self.fetch.assert_not_called()
        before = self.path.read_bytes()
        self.search.return_value = [article(1), article(3, title=" TRANSACTION   2 ")]
        repeated = self.run_workflow(2)
        self.assertEqual(repeated.attrs["processed_articles"], 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.analyze.call_count, 2)
        self.assertEqual(self.search.call_args.kwargs["search_round"], news_workflow.MAX_NEWS_EMPTY_ROUNDS - 1)
        self.assertEqual(repeated.attrs["stop_reason"], "no_new_sources")
        self.assertEqual(set(self.search.call_args.kwargs["excluded_titles"]), set(first["article_title"]))

    def test_deleted_article_is_still_skipped_using_saved_history(self):
        self.search.return_value = [article(1)]
        saved = self.run_workflow(1)
        history_before = self.history_path.read_bytes()
        storage.delete_news_rows(saved.to_dict("records"), self.path)
        repeated = self.run_workflow(1)
        self.assertTrue(repeated.empty)
        self.assertEqual(repeated.attrs["processed_articles"], 0)
        self.analyze.assert_called_once()
        self.assertEqual(self.history_path.read_bytes(), history_before)

    def test_searches_for_replacements_and_uses_excerpt_when_page_unavailable(self):
        self.search.side_effect = [
            [article(1, raw_content=None, content="")],
            [article(2, raw_content=None, content="Available search excerpt.")],
        ]
        result = self.run_workflow(1)
        self.assertEqual(result.attrs["search_rounds"], 2)
        self.assertEqual(result.iloc[0]["url"], article(2)["url"])
        self.assertEqual(self.analyze.call_args.args[0], "Available search excerpt.")
        self.assertEqual(self.fetch.call_count, 2)

    def test_page_extraction_used_when_raw_content_missing(self):
        self.search.return_value = [article(1, raw_content=None)]
        self.fetch.return_value = "<html>Source page</html>"
        with patch.object(news_workflow, "extract", return_value="Extracted full text."):
            self.run_workflow(1)
        self.assertEqual(self.analyze.call_args.args[0], "Extracted full text.")

    def test_search_excerpt_is_kept_even_when_raw_page_is_present(self):
        self.search.return_value = [article(
            1, content="Company 1 agreed to sell its subsidiary.", raw_content="Cookie settings and navigation.",
        )]
        self.run_workflow(1)
        supplied = self.analyze.call_args.args[0]
        self.assertIn("Company 1 agreed to sell its subsidiary.", supplied)
        self.assertIn("Cookie settings and navigation.", supplied)
        self.assertLess(supplied.index("Company 1"), supplied.index("Cookie settings"))
        self.fetch.assert_not_called()

    def test_sale_without_named_seller_is_excluded_and_replaced(self):
        self.search.side_effect = [[article(1)], [article(2)]]
        self.analyze.side_effect = [decision(participants=[]), decision(2)]
        result = self.run_workflow(1)
        self.assertEqual(list(result["participant_name"]), ["Company 2"])
        self.assertEqual(result.attrs["excluded_articles"], 1)
        history = storage.load_news_history(self.history_path)
        self.assertEqual(list(history["decision"]), ["excluded", "saved"])
        self.assertIn("seller-side", history.iloc[0]["reason"])
        self.search_seller_contacts.assert_called_once()

    def test_buyer_or_ambiguous_side_does_not_satisfy_seller_requirement(self):
        self.analyze.return_value = decision(participants=[])
        self.search.return_value = [article(1)]
        self.analyze.return_value = decision(participants=[
            {"name": "Buyer", "type": "company", "deal_side": "Buyer"},
            {"name": "Both", "type": "company", "deal_side": "Both"},
        ])
        result = self.run_workflow(1)
        self.assertTrue(result.empty)
        self.search_seller_contacts.assert_not_called()

    def test_legacy_rejection_can_be_reconsidered_once_under_new_rules(self):
        storage.save_news_decision(article(1), decision(qualifies=False, participants=[]), self.history_path)
        legacy = storage.load_news_history(self.history_path).drop(columns="screening_version")
        legacy.to_csv(self.history_path, index=False, encoding="utf-8-sig")
        self.search.return_value = [article(1)]
        self.assertEqual(len(self.run_workflow(1)), 1)
        history = storage.load_news_history(self.history_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["decision"], "saved")
        self.assertEqual(history.iloc[0]["screening_version"], storage.NEWS_SCREENING_VERSION)
        self.assertEqual(self.run_workflow(1).attrs["processed_articles"], 0)
        self.analyze.assert_called_once()

    def test_partial_results_survive_error_and_failed_article_can_retry(self):
        self.analyze.side_effect = [decision(description="Saved description."), RuntimeError("Simulated API failure")]
        with self.assertRaises(RuntimeError):
            self.run_workflow(2)
        self.assertEqual(len(storage.load_news_articles(self.path)), 1)
        self.analyze.side_effect = None
        self.analyze.return_value = decision(2, description="Second description.")
        self.assertEqual(len(self.run_workflow(1)), 2)
        self.assertEqual(self.analyze.call_count, 3)

    def test_rejected_article_is_only_in_history_and_skips_analysis_on_next_run(self):
        self.search.return_value = [article(1)]
        self.analyze.return_value = decision(qualifies=False, reason="A loan, not sale proceeds.", participants=[])
        first = self.run_workflow(1)
        self.assertTrue(first.empty)
        self.assertEqual(first.attrs["processed_articles"], 1)
        self.assertEqual(first.attrs["excluded_articles"], 1)
        self.assertEqual(first.attrs["saved_articles"], 0)
        history = storage.load_news_history(self.history_path)
        self.assertEqual(history.iloc[0]["decision"], "excluded")
        self.assertEqual(history.iloc[0]["reason"], "A loan, not sale proceeds.")
        self.assertFalse(self.path.exists())
        self.search.return_value = [article(1), article(3, title=" TRANSACTION   1 ")]
        self.assertEqual(self.run_workflow(1).attrs["processed_articles"], 0)
        self.analyze.assert_called_once()
        self.assertIn(article(1)["url"], self.search.call_args.kwargs["excluded_urls"])
        self.assertIn(article(1)["title"], self.search.call_args.kwargs["excluded_titles"])

    def test_mixed_decisions_save_only_eligible_articles_and_all_recipient_names(self):
        self.search.side_effect = [[article(1), article(2)], [article(3)]]
        self.analyze.side_effect = [
            decision(qualifies=False, reason="Primary investment into the company.", participants=[]),
            decision(2, participants=[{"name": "Company 2", "type": "company", "deal_side": "Seller"}, {"name": "Jane Doe", "type": "individual", "deal_side": "Seller"}]),
            decision(3),
        ]
        result = self.run_workflow(2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]["url"], article(2)["url"])
        self.assertEqual(list(result["participant_name"]), ["Company 2", "Jane Doe", "Company 3"])
        self.assertEqual(list(result["participant_type"]), ["company", "individual", "company"])
        self.assertEqual(storage.count_news_articles(result), 2)
        self.assertEqual(result.attrs["processed_articles"], 3)
        self.assertEqual(result.attrs["saved_articles"], 2)
        self.assertEqual(result.attrs["excluded_articles"], 1)
        self.assertEqual(len(storage.load_news_history(self.history_path)), 3)
        self.assertEqual(self.search.call_args_list[0].kwargs["max_results"], 2)
        self.assertEqual(self.search.call_args_list[1].kwargs["max_results"], 1)
        self.assertEqual(result.attrs["stop_reason"], "target_reached")

    def test_non_us_sale_is_journaled_and_replaced_with_us_article(self):
        self.search.side_effect = [
            [article(1, raw_content="London-based Company 1 sold its UK subsidiary.")],
            [article(2)],
        ]
        self.analyze.side_effect = deal_analyzer.analyze_news
        with patch.object(deal_analyzer, "_request_json", side_effect=[
            decision(us_scope="No", us_reason="Both businesses are based in the UK."),
            decision(2),
        ]):
            result = self.run_workflow(1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["url"], article(2)["url"])
        self.assertEqual(result.attrs["excluded_articles"], 1)
        self.assertEqual(result.attrs["saved_articles"], 1)
        history = storage.load_news_history(self.history_path)
        self.assertEqual(history.iloc[0]["decision"], "excluded")
        self.assertEqual(history.iloc[0]["us_scope"], "No")
        self.assertEqual(history.iloc[0]["us_reason"], "Both businesses are based in the UK.")
        self.assertEqual(history.iloc[1]["us_scope"], "Yes")

    def test_search_continues_beyond_five_rejections_until_target_is_saved(self):
        self.search.side_effect = [[article(i)] for i in range(1, 8)]
        self.analyze.side_effect = [
            *[decision(i, qualifies=False, participants=[]) for i in range(1, 7)],
            decision(7),
        ]
        result = self.run_workflow(1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["url"], article(7)["url"])
        self.assertEqual(result.attrs["processed_articles"], 7)
        self.assertEqual(result.attrs["excluded_articles"], 6)
        self.assertEqual(result.attrs["saved_articles"], 1)
        self.assertEqual(result.attrs["search_rounds"], 7)

    def test_empty_search_rounds_stop_and_return_saved_partial_results(self):
        self.search.side_effect = [[article(1)], *[[] for _ in range(news_workflow.MAX_NEWS_EMPTY_ROUNDS)]]
        result = self.run_workflow(2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.attrs["saved_articles"], 1)
        self.assertEqual(result.attrs["stop_reason"], "no_new_sources")
        self.assertEqual(self.search.call_count, 1 + news_workflow.MAX_NEWS_EMPTY_ROUNDS)

    def test_no_more_articles_are_analyzed_after_saved_target_is_reached(self):
        self.search.return_value = [article(1), article(2), article(3)]
        result = self.run_workflow(1)
        self.assertEqual(result.attrs["saved_articles"], 1)
        self.analyze.assert_called_once()
        self.assertEqual(len(storage.load_news_history(self.history_path)), 1)

    def test_one_article_automatically_researches_all_sellers_and_preserves_all_saved_rows(self):
        # Historical articles are displayed but do not trigger extra contact work.
        storage.save_news_article(article("old"), "Old summary.", self.path, participants=decision("old")["participants"])
        self.analyze.return_value = decision(participants=[
            {"name": "Jane Doe", "type": "individual", "organization": "Acme", "deal_side": "Seller"},
            {"name": "Seller Adviser", "type": "company", "deal_side": "Seller"},
            {"name": "Buyer", "type": "company", "deal_side": "Buyer"},
            {"name": "Both", "type": "company", "deal_side": "Both"},
        ])
        self.analyze_seller_contacts.side_effect = [
            {"email": "jane@acme.example", "linkedin": "https://linkedin.com/in/jane"},
            {field: "" for field in seller_contacts.CONTACT_FIELDS},
        ]
        result = self.run_workflow(1)
        table = seller_contacts.seller_article_table(result, storage.load_seller_contacts(self.contacts_path))
        self.assertEqual(result.attrs["saved_articles"], 1)
        self.assertEqual(result.attrs["checked_sellers"], 2)
        self.analyze.assert_called_once()
        self.assertEqual([call.args[0]["participant_name"] for call in self.search_seller_contacts.call_args_list], ["Jane Doe", "Seller Adviser"])
        self.assertEqual(list(table["participant_name"]), ["Company old", "Jane Doe", "Seller Adviser"])
        self.assertEqual(table.iloc[1]["email"], "jane@acme.example")
        self.assertEqual(table.iloc[2]["email"], "")
        self.assertEqual(table.iloc[1]["article_title"], article(1)["title"])
        self.assertEqual(table.iloc[1]["published_date"], "07 Oct 2026")
        self.assertEqual(storage.count_news_articles(table), 2)

    def test_same_seller_in_new_article_reuses_contacts_but_keeps_article_rows(self):
        self.analyze.return_value = decision(participants=[
            {"name": "Jane Doe", "type": "individual", "organization": "Acme", "deal_side": "Seller"},
        ])
        first = self.run_workflow(1)
        second = self.run_workflow(1)
        self.assertEqual(first.attrs["checked_sellers"], 1)
        self.assertEqual(second.attrs["checked_sellers"], 0)
        self.search_seller_contacts.assert_called_once()
        self.analyze_seller_contacts.assert_called_once()
        table = seller_contacts.seller_article_table(second, storage.load_seller_contacts(self.contacts_path))
        self.assertEqual(list(table["participant_name"]), ["Jane Doe", "Jane Doe"])
        self.assertEqual(storage.count_news_articles(table), 2)

    def test_contact_failure_resumes_analysis_and_skips_completed_identities(self):
        self.analyze.return_value = decision(participants=[
            {"name": "Owner", "type": "individual", "organization": "Acme", "deal_side": "Seller"},
            {"name": "Adviser", "type": "company", "deal_side": "Seller"},
        ])
        self.analyze_seller_contacts.side_effect = [{"email": "owner@acme.example"}, RuntimeError("Contact API unavailable")]
        with self.assertRaises(RuntimeError):
            self.run_workflow(1)
        self.assertEqual(len(storage.load_seller_contacts(self.contacts_path)), 1)
        self.assertTrue(storage.load_news_articles(self.path).empty)
        self.assertEqual(storage.load_news_history(self.history_path).iloc[0]["decision"], "pending_contacts")
        self.search.reset_mock()
        self.analyze_seller_contacts.side_effect = None
        self.analyze_seller_contacts.return_value = {field: "" for field in seller_contacts.CONTACT_FIELDS}
        result = self.run_workflow(1)
        self.search.assert_not_called()
        self.analyze.assert_called_once()
        self.assertEqual(self.search_seller_contacts.call_count, 3)
        self.assertEqual(result.attrs["saved_articles"], 1)
        self.assertEqual(result.attrs["checked_sellers"], 1)
        self.assertEqual(list(result["participant_name"]), ["Owner", "Adviser"])
        self.assertEqual(storage.load_news_history(self.history_path).iloc[0]["decision"], "saved")
        self.assertEqual(storage.load_seller_contacts(self.contacts_path).iloc[0]["email"], "owner@acme.example")

    def test_storage_itself_prevents_duplicates(self):
        storage.save_news_article(article(1), "Original description.", self.path)
        result = storage.save_news_article(article(2, title=" TRANSACTION 1 "), "Different description.", self.path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["description"], "Original description.")

    def test_multiple_recipient_rows_do_not_change_the_article_target(self):
        participants = [
            {"name": "Jane Doe", "type": "individual", "deal_side": "Seller"},
            {"name": "Acme", "type": "company", "deal_side": "Seller"},
            {"name": "Smith Foundation", "type": "foundation", "deal_side": "Seller"},
            {"name": "Growth Fund", "type": "fund", "deal_side": "Seller"},
        ]
        self.analyze.return_value = decision(participants=participants)
        result = self.run_workflow(2)
        self.assertEqual(len(result), 8)
        self.assertEqual(storage.count_news_articles(result), 2)
        self.assertEqual(result.attrs["saved_articles"], 2)
        self.assertEqual(self.analyze.call_count, 2)
        repeated = self.run_workflow(2)
        self.assertEqual(len(repeated), 8)
        self.assertEqual(self.analyze.call_count, 2)

    def test_storage_keeps_one_row_per_recipient_and_deduplicates_within_article(self):
        participants = [
            {"name": "Jane Doe", "type": "individual"},
            {"name": "Smith Foundation", "type": "foundation"},
            {"name": "Growth Fund", "type": "fund"},
            {"name": "  JANE  DOE ", "type": "individual"},
        ]
        result = storage.save_news_article(article(1), "Sale description.", self.path, participants=participants)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result["participant_name"]), ["Jane Doe", "Smith Foundation", "Growth Fund"])
        self.assertEqual(list(result["participant_type"]), ["individual", "foundation", "fund"])
        self.assertEqual(set(result["published_date"]), {"07 Oct 2026"})
        self.assertEqual(set(result["url"]), {article(1)["url"]})
        pd.testing.assert_frame_equal(result, storage.load_news_articles(self.path))

    def test_legacy_joined_names_expand_without_guessing_types(self):
        legacy = pd.DataFrame([{
            "article_title": "Earlier article", "published_date": "2026-10-07",
            "description": "Existing description.", "money_recipient": "Jane Doe; Acme",
            "url": article(1)["url"],
        }])
        legacy.to_csv(self.path, index=False, encoding="utf-8-sig")
        result = storage.load_news_articles(self.path)
        self.assertEqual(list(result["participant_name"]), ["Jane Doe", "Acme"])
        self.assertEqual(list(result["participant_type"]), ["", ""])
        self.assertEqual(storage.count_news_articles(result), 1)
        storage._write_table(result, self.path)
        pd.testing.assert_frame_equal(result, storage.load_news_articles(self.path))

    def test_saved_article_can_be_enriched_without_duplicates_or_losing_other_articles(self):
        storage.save_news_article(article(1), "Old summary.", self.path, participants=[{"name": "Chair", "type": "individual"}])
        storage.save_news_article(article(2), "Another article.", self.path)
        entries = [
            {"name": "Owner family", "type": "family", "role": "Seller", "deal_side": "Seller", "proceeds_recipient": "Yes"},
            {"name": "Adviser firm", "type": "company", "role": "Seller's legal counsel", "deal_side": "Seller", "proceeds_recipient": "No"},
        ]
        result = storage.save_news_article(article(1), "Updated summary.", self.path, participants=entries, replace_existing=True)
        updated = result.loc[result["url"].eq(article(1)["url"])]
        self.assertEqual(list(updated["participant_name"]), ["Owner family", "Adviser firm"])
        self.assertNotIn("proceeds_recipient", updated.columns)
        self.assertEqual(list(updated["deal_side"]), ["Seller", "Seller"])
        self.assertEqual(storage.count_news_articles(result), 2)
        self.assertEqual(result.loc[result["url"].eq(article(2)["url"])].iloc[0]["description"], "Another article.")


class NewsInterfaceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"DATABASE_URL": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def test_empty_or_no_seller_table_renders_without_contact_type_errors(self):
        empty = pd.DataFrame(columns=storage.NEWS_ARTICLE_COLUMNS)
        buyer_only = pd.DataFrame([{
            **dict.fromkeys(storage.NEWS_ARTICLE_COLUMNS, ""),
            "participant_name": "Buyer", "participant_type": "company", "deal_side": "Buyer",
            "url": "https://example.org/buyer-sale",
        }])
        for news in (empty, buyer_only):
            with self.subTest(rows=len(news)), \
                 patch("storage.load_news_articles", return_value=news), \
                 patch("storage.load_seller_contacts", return_value=pd.DataFrame(columns=storage.SELLER_CONTACT_COLUMNS)), \
                 patch("storage.load_donors", return_value=pd.DataFrame(columns=storage.COLUMNS)), \
                 patch("storage.load_funding_research", return_value=pd.DataFrame(columns=storage.FUNDING_COLUMNS)), \
                 patch("news_workflow.run_news_workflow") as search:
                app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertTrue(app.tabs[1].dataframe[0].value.empty)
                self.assertTrue(app.button(key="save_news_edits").disabled)
                self.assertFalse(app.button(key="start_news_search").disabled)
                search.assert_not_called()

    def test_saved_table_loads_without_search_and_retains_results_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "news.csv"
            storage.save_news_article(article(1), "Previously saved description.", path, participants=decision()["participants"])
            load_saved = storage.load_news_articles

            def run(article_count, on_progress):
                self.assertEqual(article_count, 3)
                on_progress("Writing descriptions...")
                frame = storage.save_news_article(article(2), "New Groq description.", path, participants=[
                    {"name": "Company 2", "type": "company", "deal_side": "Seller"},
                    {"name": "Smith Foundation", "type": "foundation", "deal_side": "Seller"},
                    {"name": "Growth Fund", "type": "fund", "deal_side": "Seller"},
                ])
                frame.attrs.update(processed_articles=5, saved_articles=1, excluded_articles=4, stop_reason="no_new_sources")
                return frame

            with patch("storage.load_news_articles", side_effect=lambda *args: load_saved(path)), patch("news_workflow.run_news_workflow", side_effect=run) as workflow, patch("main.run_workflow") as donors:
                app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual([tab.label for tab in app.tabs], ["Donor Research", "News Search"])
                self.assertEqual(len(app.tabs[1].dataframe[0].value), 1)
                workflow.assert_not_called()
                app.number_input(key="news_max_results").set_value(3).run(timeout=20)
                workflow.assert_not_called()
                app.button(key="start_news_search").click().run(timeout=20)
                self.assertFalse(app.exception)
                workflow.assert_called_once()
                donors.assert_not_called()
                self.assertEqual(len(app.tabs[1].dataframe[0].value), 4)
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[1]["description"], "New Groq description.")
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[1]["participant_name"], "Company 2")
                self.assertEqual(app.tabs[1].dataframe[0].value.iloc[2]["participant_type"], "foundation")
                self.assertIn("2 saved articles", app.tabs[1].caption[1].value)
                self.assertTrue(app.warning)
                # A new browser session also loads the saved results from disk.
                fresh = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
                self.assertEqual(len(fresh.tabs[1].dataframe[0].value), 4)
                workflow.assert_called_once()
                workflow.side_effect = RuntimeError("Simulated failure")
                with patch("logging.exception"):
                    app.button(key="start_news_search").click().run(timeout=20)
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertEqual(len(app.tabs[1].dataframe[0].value), 4)


if __name__ == "__main__":
    unittest.main()
