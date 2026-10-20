import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import storage
from seller_contacts import seller_article_table


class NewsEditingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "news.csv"
        self.cache_path = Path(directory.name) / "contacts.csv"
        for number in (1, 2):
            storage.save_news_article(
                {"title": f"Sale {number}", "url": f"https://example.org/{number}"},
                "Original description.", self.path,
                participants=[{
                    "name": "Jane Doe", "type": "individual", "organization": "Acme",
                    "role": "Owner", "deal_side": "Seller",
                }],
            )
        row = storage.load_news_articles(self.path).iloc[0].to_dict()
        self.cache = storage.save_seller_contact(row, {
            "email": "jane@acme.example", "phone": "+1 212 555 0100",
            "linkedin": "https://linkedin.com/in/jane", "contact_sources": "https://acme.example/team",
        }, "2026-10-20", self.cache_path)

    def table(self):
        return seller_article_table(storage.load_news_articles(self.path), self.cache)

    def editor(self, before):
        result = before.copy()
        result.insert(0, "_delete", False)
        return result

    def test_edits_and_intentional_blanks_persist_per_article_without_changing_cache(self):
        original = self.table()
        edited = self.editor(original)
        edited.loc[0, "description"] = "My own description."
        edited.loc[0, "participant_name"] = "Jane A. Doe"
        edited.loc[0, "email"] = "manually-added@example.org"
        edited.loc[0, "phone"] = ""
        edited.loc[0, "linkedin"] = ""
        cache_before = self.cache_path.read_bytes()
        result = storage.save_news_table_edits(original, edited, self.path)
        self.assertEqual(result.attrs["updated_rows"], 1)
        self.assertEqual(result.attrs["deleted_rows"], 0)
        table = self.table()
        self.assertEqual(table.loc[0, "participant_name"], "Jane A. Doe")
        self.assertEqual(table.loc[0, "description"], "My own description.")
        self.assertEqual(table.loc[0, "email"], "manually-added@example.org")
        self.assertEqual(table.loc[0, "phone"], "")
        self.assertEqual(table.loc[0, "linkedin"], "")
        self.assertEqual(table.loc[1, "email"], "jane@acme.example")
        self.assertEqual(table.loc[1, "phone"], "+1 212 555 0100")
        self.assertNotIn("contact_overrides", table.columns)
        self.assertEqual(self.cache_path.read_bytes(), cache_before)

    def test_edit_and_delete_together_preserve_rows_added_after_page_loaded(self):
        original = self.table()
        edited = self.editor(original)
        edited.loc[0, "role"] = "Former owner"
        edited.loc[1, "_delete"] = True
        storage.save_news_article(
            {"title": "New sale", "url": "https://example.org/new"}, "New description.", self.path,
            participants=[{"name": "Other seller", "type": "company", "deal_side": "Seller"}],
        )
        result = storage.save_news_table_edits(original, edited, self.path)
        self.assertEqual(result.attrs["updated_rows"], 1)
        self.assertEqual(result.attrs["deleted_rows"], 1)
        self.assertEqual(list(result["participant_name"]), ["Jane Doe", "Other seller"])
        self.assertEqual(result.iloc[0]["role"], "Former owner")

    def test_stale_edit_fails_without_overwriting_any_rows(self):
        original = self.table()
        edited = self.editor(original)
        edited.loc[0, "description"] = "Wanted edit."
        edited.loc[1, "_delete"] = True
        storage.delete_news_rows([original.iloc[1].to_dict()], self.path)
        before_bytes = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "changed or deleted"):
            storage.save_news_table_edits(original, edited, self.path)
        self.assertEqual(self.path.read_bytes(), before_bytes)

    def test_manual_contacts_survive_later_discovery_and_do_not_leak_into_new_articles(self):
        original = self.table()
        edited = self.editor(original)
        edited.loc[0, "email"] = "my-note@example.org"
        storage.save_news_table_edits(original, edited, self.path)
        result = storage.save_news_article(
            {"title": "Third sale", "url": "https://example.org/3"}, "Third summary.", self.path,
            participants=[{"name": "Jane Doe", "type": "individual", "organization": "Acme", "deal_side": "Seller"}],
        )
        # Also exercise the immediate in-memory result, whose extra fields may be NaN.
        table = seller_article_table(result, self.cache)
        self.assertEqual(list(table["email"]), ["my-note@example.org", "jane@acme.example", "jane@acme.example"])
        pd.testing.assert_frame_equal(table, self.table())

    def test_delete_all_keeps_readable_headers_and_no_changes_do_not_rewrite_file(self):
        original = self.table()
        edited = self.editor(original)
        before_bytes = self.path.read_bytes()
        result = storage.save_news_table_edits(original, edited, self.path)
        self.assertEqual(result.attrs["updated_rows"], 0)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        edited["_delete"] = True
        storage.save_news_table_edits(original, edited, self.path)
        self.assertTrue(self.table().empty)
        self.assertEqual(list(storage.load_news_articles(self.path).columns), storage.NEWS_ARTICLE_COLUMNS)

    def test_streamlit_saves_cell_edits_and_reload_reads_them_without_search(self):
        load_saved = storage.load_news_articles
        save_edits = storage.save_news_table_edits
        native_editor = st.data_editor
        state = {"edit": True}

        def editor(data, **kwargs):
            edited = native_editor(data, **kwargs)
            if state["edit"]:
                edited.loc[0, "description"] = "Description entered in Streamlit."
                edited.loc[0, "email"] = "manual@example.org"
            return edited

        def save(before, after):
            result = save_edits(before, after, self.path)
            state["edit"] = False
            return result

        with patch("storage.load_news_articles", side_effect=lambda *args: load_saved(self.path)), \
             patch("storage.load_seller_contacts", return_value=self.cache), \
             patch("storage.save_news_table_edits", side_effect=save) as save_mock, \
             patch.object(st, "data_editor", side_effect=editor), \
             patch("news_workflow.run_news_workflow") as search:
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
            self.assertFalse(app.exception)
            save_mock.assert_not_called()
            app.button(key="save_news_edits").click().run(timeout=20)
            self.assertFalse(app.exception)
            save_mock.assert_called_once()
            self.assertEqual(app.tabs[1].dataframe[0].value.loc[0, "email"], "manual@example.org")
            fresh = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.tabs[1].dataframe[0].value.loc[0, "description"], "Description entered in Streamlit.")
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
