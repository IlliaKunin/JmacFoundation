import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import storage


class NewsDeletionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"DATABASE_URL": ""})
        environment.start()
        self.addCleanup(environment.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "news.csv"
        self.article = {"title": "Test sale", "url": "https://example.org/sale"}
        storage.save_news_article(self.article, "Sale summary.", self.path, participants=[
            {"name": "Owner", "type": "individual", "deal_side": "Seller"},
            {"name": "Adviser", "type": "company", "deal_side": "Seller"},
        ])

    def test_deletes_only_selected_row_and_persists(self):
        original = storage.load_news_articles(self.path)
        result = storage.delete_news_rows([original.iloc[0].to_dict()], self.path)
        self.assertEqual(result.attrs["deleted_rows"], 1)
        self.assertEqual(list(result["participant_name"]), ["Adviser"])
        self.assertEqual(list(storage.load_news_articles(self.path)["participant_name"]), ["Adviser"])

    def test_stale_positions_do_not_delete_different_records(self):
        original = storage.load_news_articles(self.path)
        selected = original.iloc[0].to_dict()
        storage._write_table(original.iloc[::-1].reset_index(drop=True), self.path)
        result = storage.delete_news_rows([selected], self.path)
        self.assertEqual(list(result["participant_name"]), ["Adviser"])
        # Selecting an old record a second time leaves the surviving row alone.
        result = storage.delete_news_rows([selected], self.path)
        self.assertEqual(result.attrs["deleted_rows"], 0)
        self.assertEqual(list(result["participant_name"]), ["Adviser"])

    def test_delete_all_preserves_csv_headers(self):
        original = storage.load_news_articles(self.path)
        result = storage.delete_news_rows(original.to_dict("records"), self.path)
        self.assertEqual(result.attrs["deleted_rows"], 2)
        self.assertTrue(storage.load_news_articles(self.path).empty)
        self.assertEqual(list(storage.load_news_articles(self.path).columns), storage.NEWS_ARTICLE_COLUMNS)

    def test_interface_selects_multiple_rows_and_deletes_without_search(self):
        storage.save_news_article({"title": "Another sale", "url": "https://example.org/other"}, "Other summary.", self.path, participants=[
            {"name": "Another seller", "type": "company", "deal_side": "Seller"},
        ])
        original_editor = st.data_editor
        original_load = storage.load_news_articles
        original_save = storage.save_news_table_edits
        state = {"select": True}

        def editor(data, **kwargs):
            result = original_editor(data, **kwargs)
            if str(kwargs.get("key", "")).startswith("news_table_") and state["select"]:
                result.loc[[0, 2], "_delete"] = True
            return result

        def save(before, after):
            state["select"] = False
            return original_save(before, after, self.path)

        with patch.object(storage, "load_news_articles", side_effect=lambda *args: original_load(self.path)), patch.object(storage, "save_news_table_edits", side_effect=save) as remove, patch.object(storage, "load_donors", return_value=pd.DataFrame(columns=storage.COLUMNS)), patch.object(storage, "load_funding_research", return_value=pd.DataFrame(columns=storage.FUNDING_COLUMNS)), patch.object(st, "data_editor", side_effect=editor), patch("main.run_workflow") as donors, patch("news_workflow.run_news_workflow") as news:
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="save_news_edits").disabled)
            app.button(key="save_news_edits").click().run(timeout=20)
            self.assertFalse(app.exception)
            remove.assert_called_once()
            self.assertEqual(list(app.tabs[1].dataframe[0].value["participant_name"]), ["Adviser"])
            self.assertFalse(app.button(key="save_news_edits").disabled)
            self.assertIn("Saved changes to 0 rows. Deleted 2 rows.", [message.value for message in app.success])
            news.assert_not_called()
            donors.assert_not_called()
            fresh = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
            self.assertEqual(list(fresh.tabs[1].dataframe[0].value["participant_name"]), ["Adviser"])


if __name__ == "__main__":
    unittest.main()
