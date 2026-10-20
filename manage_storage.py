"""Explicit one-time import and Excel-compatible export of shared data."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from shared_storage import import_local_documents, uses_database
from seller_contacts import seller_article_table
from storage import NEWS_ARTICLES_FILE, load_news_articles, load_seller_contacts


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["import", "export"])
    parser.add_argument("--output", type=Path, default=Path("exports/sellers.csv"))
    args = parser.parse_args()
    if args.action == "import":
        imported = import_local_documents()
        print(f"Imported {len(imported)} documents. Existing shared documents were kept.")
        for name in imported:
            print(name)
    else:
        if not uses_database(NEWS_ARTICLES_FILE):
            parser.error("Set DATABASE_URL to export the shared table.")
        table = seller_article_table(load_news_articles(), load_seller_contacts())
        columns = [
            "participant_name", "participant_type", "role", "email", "phone",
            "contact_page", "linkedin", "url", "published_date",
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table[columns].to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Exported {len(table)} rows to {args.output}.")


if __name__ == "__main__":
    main()
