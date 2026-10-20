from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import shared_storage
import storage
from seller_contacts import seller_article_table


class SharedStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.news_path = self.root / "news_articles.csv"
        self.history_path = self.root / "news_article_history.csv"
        self.contacts_path = self.root / "seller_contacts.csv"
        for patcher in (
            patch.object(shared_storage, "DATA_ROOT", self.root),
            patch.dict(os.environ, {"DATABASE_URL": "sqlite:///" + str(self.root / "shared.db")}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def article(self, number):
        return {"title": f"Sale {number}", "url": f"https://example.org/{number}", "published_date": "2026-10-07"}

    def save(self, number):
        return storage.save_news_article(self.article(number), "Sale summary.", self.news_path, participants=[
            {"name": f"Seller {number}", "type": "company", "deal_side": "Seller"},
        ])

    def table(self):
        return seller_article_table(storage.load_news_articles(self.news_path), storage.load_seller_contacts(self.contacts_path))

    def test_concurrent_users_append_without_lost_rows_and_no_local_csv_is_written(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.save, range(24)))
        result = storage.load_news_articles(self.news_path)
        self.assertEqual(len(result), 24)
        self.assertEqual(storage.count_news_articles(result), 24)
        self.assertFalse(self.news_path.exists())
        # Opening a fresh connection is what a restarted app/another user does.
        self.assertEqual(len(storage.load_news_articles(self.news_path)), 24)

    def test_concurrent_duplicate_articles_and_contacts_are_stored_once(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(self.save, [1] * 12))
        self.assertEqual(sum(frame.attrs["article_inserted"] for frame in results), 1)
        self.assertEqual(len(storage.load_news_articles(self.news_path)), 1)
        row = storage.load_news_articles(self.news_path).iloc[0].to_dict()
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: storage.save_seller_contact(row, {"email": "hello@example.org"}, "2026-10-20", self.contacts_path), range(12)))
        self.assertEqual(len(storage.load_seller_contacts(self.contacts_path)), 1)

    def test_edit_and_delete_are_shared_and_preserve_new_rows(self):
        self.save(1)
        self.save(2)
        before = self.table()
        edited = before.copy()
        edited["_delete"] = False
        edited.loc[0, "email"] = "manual@example.org"
        edited.loc[1, "_delete"] = True
        self.save(3)
        storage.save_news_table_edits(before, edited, self.news_path)
        fresh = self.table()
        self.assertEqual(list(fresh["participant_name"]), ["Seller 1", "Seller 3"])
        self.assertEqual(fresh.iloc[0]["email"], "manual@example.org")
        self.assertFalse(self.news_path.exists())

    def test_two_users_editing_only_contacts_get_a_conflict_instead_of_overwriting(self):
        self.save(1)
        before = self.table()
        first = before.copy()
        first.loc[0, "email"] = "first@example.org"
        storage.save_news_table_edits(before, first, self.news_path)
        second = before.copy()
        second.loc[0, "email"] = "second@example.org"
        with self.assertRaisesRegex(ValueError, "another user"):
            storage.save_news_table_edits(before, second, self.news_path)
        self.assertEqual(self.table().iloc[0]["email"], "first@example.org")

    def test_multi_document_transaction_rolls_back_on_error(self):
        with self.assertRaisesRegex(RuntimeError, "Abort"):
            with shared_storage.storage_transaction(self.news_path):
                self.save(1)
                shared_storage.write_text(self.history_path, "pending source")
                raise RuntimeError("Abort")
        self.assertTrue(storage.load_news_articles(self.news_path).empty)
        self.assertIsNone(shared_storage.read_text(self.history_path))

    def test_import_is_explicit_idempotent_and_never_restores_deleted_shared_data(self):
        self.news_path.write_text("article_title,url\nLegacy sale,https://example.org/legacy\n", encoding="utf-8")
        self.assertTrue(storage.load_news_articles(self.news_path).empty)
        self.assertEqual(shared_storage.import_local_documents(), ["news_articles.csv"])
        self.assertEqual(len(storage.load_news_articles(self.news_path)), 1)
        before = storage.load_news_articles(self.news_path)
        storage.delete_news_rows(before.to_dict("records"), self.news_path)
        self.assertEqual(shared_storage.import_local_documents(), [])
        self.assertTrue(storage.load_news_articles(self.news_path).empty)
        self.assertIn("Legacy sale", self.news_path.read_text(encoding="utf-8"))

    def test_all_donor_outputs_use_shared_storage_including_json_details(self):
        donors_path = self.root / "donors.csv"
        sources_path = self.root / "processed_articles.csv"
        funding_path = self.root / "funding_research.csv"
        details_path = self.root / "funding_research.json"
        result = {"donors": [{"name": "Test Foundation", "type": "foundation"}]}
        storage.save_donors(self.article(1), "Source text.", result, donors_path)
        storage.save_processed_article(self.article(1), sources_path, donors_path)
        funding = {"name": "Test Foundation", "funding_match": "Yes", "applications": "Open", "contact_method": "https://example.org/apply"}
        storage.save_funding_research(funding, funding_path, details_path)
        self.assertEqual(len(storage.load_donors(donors_path)), 1)
        self.assertEqual(len(storage.load_processed_articles(sources_path, donors_path)), 1)
        self.assertEqual(storage.load_funding_research(funding_path).iloc[0]["Name"], "Test Foundation")
        self.assertEqual(storage.load_funding_details(details_path)[0]["name"], "Test Foundation")
        self.assertFalse(any(path.exists() for path in (donors_path, sources_path, funding_path, details_path)))

    def test_unavailable_database_does_not_fall_back_to_bundled_csv(self):
        self.news_path.write_text("article_title,url\nStale data,https://example.org/old\n", encoding="utf-8")
        with patch.object(shared_storage.sqlite3, "connect", side_effect=sqlite3.OperationalError("Unavailable")):
            with self.assertRaises(sqlite3.OperationalError):
                storage.load_news_articles(self.news_path)

    def test_paths_outside_data_root_keep_their_local_behavior(self):
        with tempfile.TemporaryDirectory() as other:
            path = Path(other) / "news_articles.csv"
            storage.save_news_article(self.article(1), "Local test.", path)
            self.assertTrue(path.exists())


class PostgreSQLAdapterTests(unittest.TestCase):
    def test_postgresql_uses_transaction_lock_parameterized_queries_commit_and_close(self):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = ("csv contents",)
        driver = types.SimpleNamespace(connect=MagicMock(return_value=connection))
        path = shared_storage.DATA_ROOT / "news_articles.csv"
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test-host/test-db"}), patch.dict("sys.modules", {"psycopg": driver}):
            with shared_storage.storage_transaction(path):
                self.assertEqual(shared_storage.read_text(path), "csv contents")
                shared_storage.write_text(path, "new content")
        driver.connect.assert_called_once_with("postgresql://test-host/test-db", connect_timeout=15)
        statements = [call.args[0] for call in connection.execute.call_args_list]
        self.assertIn("SELECT pg_advisory_xact_lock(%s)", statements)
        self.assertIn("SELECT content FROM research_documents WHERE name = %s", statements)
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()
        connection.close.assert_called_once()

    def test_postgresql_rolls_back_and_closes_on_failure(self):
        connection = MagicMock()
        driver = types.SimpleNamespace(connect=MagicMock(return_value=connection))
        path = shared_storage.DATA_ROOT / "news_articles.csv"
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test-host/test-db"}), patch.dict("sys.modules", {"psycopg": driver}):
            with self.assertRaises(ValueError):
                with shared_storage.storage_transaction(path):
                    raise ValueError("Abort")
        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
