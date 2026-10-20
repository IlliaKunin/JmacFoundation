from pathlib import Path
import json
import unicodedata
import hashlib
from collections import Counter
from io import StringIO

import pandas as pd
from shared_storage import read_text, write_text, transactional


DATA_FILE = Path(__file__).with_name("donors.csv")
FUNDING_DATA_FILE = Path(__file__).with_name("funding_research.csv")
FUNDING_JSON_FILE = Path(__file__).with_name("funding_research.json")
PROCESSED_ARTICLES_FILE = Path(__file__).with_name("processed_articles.csv")
ARTICLE_COLUMNS = ["url", "article_title"]
EVENTS_FILE = Path(__file__).with_name("events.csv")
EVENT_PEOPLE_FILE = Path(__file__).with_name("event_people.csv")
EVENT_ARTICLES_FILE = Path(__file__).with_name("processed_event_articles.csv")
NEWS_ARTICLES_FILE = Path(__file__).with_name("news_articles.csv")
NEWS_ARTICLE_COLUMNS = [
    "article_title", "published_date", "description", "participant_name", "participant_type",
    "organization", "role", "deal_side", "url",
]
SELLER_CONTACTS_FILE = Path(__file__).with_name("seller_contacts.csv")
SELLER_CONTACT_FIELDS = ("email", "phone", "contact_page", "linkedin", "contact_sources")
SELLER_CONTACT_COLUMNS = [
    "contact_key", "participant_name", "participant_type", "organization", "role",
    "email", "phone", "contact_page", "linkedin", "contact_sources", "checked_at",
]
NEWS_HISTORY_FILE = Path(__file__).with_name("news_article_history.csv")
NEWS_SCREENING_VERSION = "us_sales_v3"
NEWS_HISTORY_COLUMNS = [
    "url", "article_title", "published_date", "decision", "reason", "description", "money_recipient",
    "screening_version",
    "us_scope", "us_reason",
    "participants",
]
EVENT_COLUMNS = [
    "event_id", "event_name", "event_type", "description", "participants",
    "published_date", "article_title", "article_url", "source_excerpt", "research_complete",
]
EVENT_PEOPLE_COLUMNS = [
    "event_id", "event_name", "article_description", "person_name", "company",
    "position", "event_connection", "email", "phone", "linkedin", "source_urls",
]
FUNDING_COLUMNS = ["Name", "Funding Match", "Applications", "Contact Method"]
COLUMNS = [
    "published_date",
    "article_summary",
    "donor_name",
    "donor_type",
    "email",
    "phone",
    "contact_page",
    "article_title",
    "url",
]


def format_publication_date(value):
    if not value:
        return value
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return value
    return date.strftime("%d %b %Y")


def load_donors(path=DATA_FILE):
    df = _load_table(path, COLUMNS)
    df["published_date"] = df["published_date"].map(format_publication_date)
    return df


def has_url(df, url):
    return "url" in df.columns and df["url"].eq(url).any()


def normalize_title(title):
    if title is None or pd.isna(title):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(title)).split()).casefold()


def unique_donor_names(df):
    names = {}
    for name in df["donor_name"]:
        key = normalize_title(name)
        if key:
            names.setdefault(key, str(name).strip())
    return list(names.values())


def has_article(df, article):
    if has_url(df, article["url"]):
        return True
    title = normalize_title(article.get("title"))
    return bool(title) and "article_title" in df.columns and df["article_title"].map(normalize_title).eq(title).any()


def load_processed_articles(path=PROCESSED_ARTICLES_FILE, donors_path=DATA_FILE):
    history = _load_table(path, ARTICLE_COLUMNS)
    # Existing donor rows also prove their source has already been analyzed.
    donors = load_donors(donors_path)
    legacy_sources = donors.loc[donors["url"].ne(""), ARTICLE_COLUMNS]
    history = pd.concat([history, legacy_sources], ignore_index=True)
    return history.loc[history["url"].ne("")].drop_duplicates(subset="url").reset_index(drop=True)


@transactional
def save_processed_article(article, path=PROCESSED_ARTICLES_FILE, donors_path=DATA_FILE):
    path = Path(path)
    history = load_processed_articles(path, donors_path)
    if not has_article(history, article):
        row = pd.DataFrame([{
            "url": article["url"],
            "article_title": article.get("title") or "",
        }], columns=ARTICLE_COLUMNS)
        history = pd.concat([history, row], ignore_index=True)
    _write_table(history, path)
    return history


def has_research(df, name):
    key = normalize_title(name)
    return bool(key) and df["Name"].map(normalize_title).eq(key).any()


def clean_contact(value):
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.casefold() in {"n/a", "none", "null"} else value


@transactional
def save_donors(article, article_text, result, path=DATA_FILE, max_new_donors=None):
    if max_new_donors is not None and (
        not isinstance(max_new_donors, int) or max_new_donors < 0
    ):
        raise ValueError("max_new_donors must be a non-negative integer.")
    path = Path(path)
    df = load_donors(path)
    if max_new_donors == 0 or has_article(df, article):
        return df

    summary = " ".join((article.get("content") or article_text or "").split())
    if len(summary) > 500:
        summary = summary[:497] + "..."
    rows = []
    seen_donors = {normalize_title(name) for name in unique_donor_names(df)}
    for donor in result.get("donors", []):
        name = (donor.get("name") or "").strip()
        donor_key = normalize_title(name)
        if not name or donor_key in seen_donors:
            continue
        seen_donors.add(donor_key)
        contacts = donor.get("contact_information") or {}
        rows.append({
            "published_date": format_publication_date(article.get("published_date")) or "",
            "article_summary": summary,
            "donor_name": name,
            "donor_type": donor.get("type", ""),
            "email": clean_contact(contacts.get("email")),
            "phone": clean_contact(contacts.get("phone")),
            "contact_page": clean_contact(contacts.get("contact_page")),
            "article_title": article.get("title") or "",
            "url": article["url"],
        })
        if max_new_donors is not None and len(rows) >= max_new_donors:
            break

    if rows:
        new_rows = pd.DataFrame(rows, columns=COLUMNS)
        df = pd.concat([df, new_rows], ignore_index=True) if not df.empty else new_rows

    _write_table(df, path)

    return df


def load_funding_research(path=FUNDING_DATA_FILE):
    return _load_table(path, FUNDING_COLUMNS)


def load_funding_details(path=FUNDING_JSON_FILE):
    content = read_text(path)
    return json.loads(content)["donors"] if content is not None else []


@transactional
def save_funding_research(result, path=FUNDING_DATA_FILE, details_path=FUNDING_JSON_FILE):
    path = Path(path)
    details_path = Path(details_path)
    name = (result.get("name") or "").strip()
    if not name:
        raise ValueError("Funding research requires a donor name.")
    if result.get("funding_match") not in {"Yes", "No", "Unknown"}:
        raise ValueError("Invalid Funding Match value.")
    if result.get("applications") not in {"Open", "Closed", "Invitation Only", "Unknown"}:
        raise ValueError("Invalid Applications value.")

    df = load_funding_research(path)
    key = normalize_title(name)
    if has_research(df, name):
        return df
    row = pd.DataFrame([{
        "Name": name,
        "Funding Match": result["funding_match"],
        "Applications": result["applications"],
        "Contact Method": clean_contact(result.get("contact_method")),
    }], columns=FUNDING_COLUMNS)
    df = pd.concat([df, row], ignore_index=True) if not df.empty else row

    details = [
        item for item in load_funding_details(details_path)
        if normalize_title(item.get("name")) != key
    ]
    details.append({**result, "name": name})
    write_text(details_path, json.dumps({"donors": details}, indent=2, ensure_ascii=False))
    _write_table(df, path)
    return df


def _load_table(path, columns):
    content = read_text(path)
    if content is None:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(StringIO(content), dtype=str, keep_default_na=False)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def _write_table(df, path):
    write_text(path, df.to_csv(index=False))


def load_news_articles(path=NEWS_ARTICLES_FILE):
    df = _load_table(path, [])
    for old, new in (("money_recipient", "participant_name"), ("recipient_type", "participant_type")):
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    for column in NEWS_ARTICLE_COLUMNS:
        if column not in df.columns:
            df[column] = "Unknown" if column in {"deal_side", "proceeds_recipient"} else ""
    df = df.drop(columns="proceeds_recipient", errors="ignore")
    organizational = ~df["participant_type"].isin(["individual", "family", ""])
    missing_organization = organizational & df["organization"].eq("")
    df.loc[missing_organization, "organization"] = df.loc[missing_organization, "participant_name"]
    df["published_date"] = df["published_date"].map(format_publication_date)
    # Earlier versions joined names using '; '. Expand those legacy rows without
    # guessing recipient types. New rows with a known type preserve names intact.
    df["participant_name"] = [
        ([name.strip() for name in value.split("; ") if name.strip()] or [""])
        if not kind else [value]
        for value, kind in zip(df["participant_name"], df["participant_type"])
    ]
    df = df.explode("participant_name", ignore_index=True)
    return df[NEWS_ARTICLE_COLUMNS + [column for column in df.columns if column not in NEWS_ARTICLE_COLUMNS]]


def count_news_articles(df):
    return int(df.loc[df["url"].ne(""), "url"].nunique())


def news_row_revision(row):
    values = [
        "" if pd.isna(row.get(field, "")) else str(row.get(field, ""))
        for field in [*NEWS_ARTICLE_COLUMNS, "contact_overrides"]
    ]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


@transactional
def delete_news_rows(rows, path=NEWS_ARTICLES_FILE):
    """Delete selected row snapshots, preserving other rows and search history."""
    df = load_news_articles(path)
    # Compare records, not positions: another run may have appended or reordered
    # the file since the user selected rows in the browser.
    def identity(row):
        return tuple(str(row.get(column, "")) for column in NEWS_ARTICLE_COLUMNS)

    pending = Counter(identity(row) for row in rows)
    keep = []
    deleted = 0
    for row in df.to_dict("records"):
        key = identity(row)
        if pending[key] > 0:
            pending[key] -= 1
            deleted += 1
            keep.append(False)
        else:
            keep.append(True)
    if deleted:
        df = df.loc[keep].reset_index(drop=True)
        _write_table(df, path)
    df.attrs["deleted_rows"] = deleted
    return df


@transactional
def save_news_table_edits(original, edited, path=NEWS_ARTICLES_FILE):
    """Apply edits/deletions to row snapshots without replacing unseen records."""
    fields = [*NEWS_ARTICLE_COLUMNS, *SELLER_CONTACT_FIELDS]
    if len(original) != len(edited) or any(field not in edited for field in fields):
        raise ValueError("The table has changed. Reload it before saving.")

    def text(value):
        return "" if value is None or pd.isna(value) else str(value)

    def identity(row):
        return tuple(text(row.get(field, "")) for field in NEWS_ARTICLE_COLUMNS)

    df = load_news_articles(path)
    records = df.to_dict("records")
    positions = {}
    for index, row in enumerate(records):
        positions.setdefault(identity(row), []).append(index)
    updates, removed = {}, set()
    for before, after in zip(original.to_dict("records"), edited.to_dict("records")):
        delete = bool(after.get("_delete", False))
        changed = any(text(before.get(field)) != text(after.get(field)) for field in fields)
        if not delete and not changed:
            continue
        matches = positions.get(identity(before), [])
        if not matches:
            raise ValueError("A selected row was changed or deleted elsewhere. Reload the page before saving.")
        index = matches.pop(0)
        if before.get("_row_revision") and before["_row_revision"] != news_row_revision(records[index]):
            raise ValueError("This row was edited by another user. Reload the page before saving.")
        if delete:
            removed.add(index)
            continue
        if not text(after["participant_name"]).strip():
            raise ValueError("A person / organization name cannot be empty. Use Delete to remove the row.")
        record = {**records[index], **{field: text(after[field]) for field in NEWS_ARTICLE_COLUMNS}}
        # Per-article overrides preserve intentional blanks and do not alter
        # another article's contacts when the same identity appears there.
        record["contact_overrides"] = json.dumps(
            {field: text(after[field]) for field in SELLER_CONTACT_FIELDS}, ensure_ascii=False,
        )
        updates[index] = record
    if updates or removed:
        columns = list(df.columns)
        if updates and "contact_overrides" not in columns:
            columns.append("contact_overrides")
        df = pd.DataFrame([
            updates.get(index, row) for index, row in enumerate(records) if index not in removed
        ], columns=columns)
        df = df.fillna("")
        _write_table(df, path)
    df.attrs.update(updated_rows=len(updates), deleted_rows=len(removed))
    return df


@transactional
def save_news_article(article, description, path=NEWS_ARTICLES_FILE, money_recipient="", recipients=None, participants=None, replace_existing=False):
    if not isinstance(description, str) or not description.strip():
        raise ValueError("News analysis requires a nonempty description.")
    df = load_news_articles(path)
    inserted = False
    if not has_article(df, article) or replace_existing:
        if replace_existing:
            matches = df["url"].eq(article["url"])
            title_key = normalize_title(article.get("title"))
            if title_key:
                matches |= df["article_title"].map(normalize_title).eq(title_key)
            df = df.loc[~matches]
        if participants is not None:
            recipients = participants
        if recipients is None:
            recipients = [{"name": name.strip(), "type": ""} for name in money_recipient.split("; ") if name.strip()]
        if not isinstance(recipients, list):
            raise ValueError("News recipients must be a list.")
        names, seen = [], set()
        for recipient in recipients:
            if not isinstance(recipient, dict) or not isinstance(recipient.get("name"), str):
                raise ValueError("Each recipient requires a name.")
            name = recipient["name"].strip()
            key = normalize_title(name)
            if not key or key in seen:
                continue
            seen.add(key)
            kind = recipient.get("type", "")
            if kind not in {"individual", "company", "family", "foundation", "fund", "organization", ""}:
                raise ValueError("Invalid news participant type.")
            role = recipient.get("role", "")
            side = recipient.get("deal_side", "Unknown")
            proceeds = recipient.get("proceeds_recipient", "Unknown")
            organization = recipient.get("organization", "")
            if not isinstance(organization, str):
                organization = ""
            if not organization and kind not in {"individual", "family", ""}:
                organization = name
            names.append({
                "name": name, "type": kind, "role": role if isinstance(role, str) else "",
                "organization": organization,
                "deal_side": side if side in {"Seller", "Buyer", "Both", "Other", "Unknown"} else "Unknown",
                "proceeds_recipient": proceeds if proceeds in {"Yes", "No", "Unknown"} else "Unknown",
            })
        base = {
            "article_title": article.get("title") or "",
            "published_date": format_publication_date(article.get("published_date")) or "",
            "description": description.strip(),
            "url": article["url"],
        }
        # Keep an article with an unidentified recipient visible, as before.
        row = pd.DataFrame([
            {
                **base, "participant_name": participant["name"], "participant_type": participant["type"],
                "organization": participant["organization"],
                "role": participant["role"], "deal_side": participant["deal_side"],
            }
            for participant in (names or [{"name": "", "type": "", "organization": "", "role": "", "deal_side": "Unknown"}])
        ], columns=NEWS_ARTICLE_COLUMNS)
        df = pd.concat([df, row], ignore_index=True) if not df.empty else row
        _write_table(df, path)
        inserted = True
    df.attrs["article_inserted"] = inserted
    return df


def seller_contact_key(row):
    name = normalize_title(row.get("participant_name"))
    organization = normalize_title(row.get("organization"))
    # Preserve separate identities when a person's/family's affiliation is unknown.
    scope = organization or (row.get("url") or "")
    identity = [name, normalize_title(row.get("participant_type")), scope]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def load_seller_contacts(path=SELLER_CONTACTS_FILE):
    return _load_table(path, SELLER_CONTACT_COLUMNS)


@transactional
def save_seller_contact(row, contacts, checked_at, path=SELLER_CONTACTS_FILE):
    df = load_seller_contacts(path)
    key = seller_contact_key(row)
    if key in set(df["contact_key"]):
        return df
    record = {
        "contact_key": key, "participant_name": row["participant_name"],
        "participant_type": row.get("participant_type", ""), "organization": row.get("organization", ""),
        "role": row.get("role", ""), "checked_at": checked_at,
        **{field: contacts.get(field, "") for field in ("email", "phone", "contact_page", "linkedin", "contact_sources")},
    }
    df = pd.concat([df, pd.DataFrame([record], columns=SELLER_CONTACT_COLUMNS)], ignore_index=True)
    _write_table(df, path)
    return df


def load_news_history(path=NEWS_HISTORY_FILE):
    return _load_table(path, NEWS_HISTORY_COLUMNS)


@transactional
def save_news_decision(article, analysis, path=NEWS_HISTORY_FILE):
    df = load_news_history(path)
    # Replace an older screening decision for the same source after a rule change.
    matches = df["url"].eq(article["url"])
    title_key = normalize_title(article.get("title"))
    if title_key:
        matches |= df["article_title"].map(normalize_title).eq(title_key)
    df = df.loc[~matches]
    row = pd.DataFrame([{
        "url": article["url"],
        "article_title": article.get("title") or "",
        "published_date": format_publication_date(article.get("published_date")) or "",
        "decision": (
            "pending_contacts" if analysis.get("contact_research_pending")
            else "saved" if analysis["qualifies"] else "excluded"
        ),
        "reason": analysis["reason"],
        "description": analysis["description"],
        "money_recipient": "; ".join(person["name"] for person in analysis["participants"] if person.get("proceeds_recipient") == "Yes"),
        "participants": json.dumps(analysis["participants"], ensure_ascii=False),
        "screening_version": NEWS_SCREENING_VERSION,
        "us_scope": analysis.get("us_scope", "Unknown"),
        "us_reason": analysis.get("us_reason", ""),
    }], columns=NEWS_HISTORY_COLUMNS)
    df = pd.concat([df, row], ignore_index=True) if not df.empty else row
    _write_table(df, path)
    return df


def load_events(path=EVENTS_FILE):
    return _load_table(path, EVENT_COLUMNS)


def load_event_people(path=EVENT_PEOPLE_FILE):
    return _load_table(path, EVENT_PEOPLE_COLUMNS)


def load_event_articles(path=EVENT_ARTICLES_FILE):
    return _load_table(path, ARTICLE_COLUMNS)


@transactional
def save_event_article(article, path=EVENT_ARTICLES_FILE):
    df = load_event_articles(path)
    if not has_article(df, article):
        row = pd.DataFrame([{"url": article["url"], "article_title": article.get("title") or ""}])
        df = pd.concat([df, row], ignore_index=True)
        _write_table(df, path)
    return df


@transactional
def save_events(article, events, path=EVENTS_FILE):
    df = load_events(path)
    known_ids = set(df["event_id"])
    rows = []
    for event in events:
        organizations = sorted({
            normalize_title(participant["name"])
            for participant in event["participants"] if participant["entity_type"] == "organization"
        })
        identity = json.dumps([normalize_title(event["event_name"]), event["event_type"], organizations])
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        if event_id in known_ids:
            continue
        known_ids.add(event_id)
        rows.append({
            "event_id": event_id,
            "event_name": event["event_name"], "event_type": event["event_type"],
            "description": event["description"],
            "participants": json.dumps(event["participants"], ensure_ascii=False),
            "published_date": format_publication_date(article.get("published_date")) or "",
            "article_title": article.get("title") or "", "article_url": article["url"],
            "source_excerpt": event.get("source_excerpt", ""),
            "research_complete": "no",
        })
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=EVENT_COLUMNS)], ignore_index=True)
        _write_table(df, path)
    return df


@transactional
def save_event_people(event, people, path=EVENT_PEOPLE_FILE):
    df = load_event_people(path)
    records = df.to_dict("records")
    known = {
        (row["event_id"], normalize_title(row["person_name"]), normalize_title(row["company"])): row
        for row in records
    }
    changed = False
    for person in people:
        key = (event["event_id"], normalize_title(person["name"]), normalize_title(person.get("company")))
        row = {
            "event_id": event["event_id"], "event_name": event["event_name"],
            "article_description": event["description"], "person_name": person["name"],
            "company": person.get("company", ""), "position": person.get("position", ""),
            "event_connection": person["event_connection"],
            "email": clean_contact(person.get("email")), "phone": clean_contact(person.get("phone")),
            "linkedin": clean_contact(person.get("linkedin")),
            "source_urls": "\n".join(dict.fromkeys(person["source_urls"])),
        }
        if key in known:
            existing = known[key]
            for field in ("position", "event_connection", "email", "phone", "linkedin"):
                if not existing[field] and row[field]:
                    existing[field] = row[field]
                    changed = True
            merged = "\n".join(dict.fromkeys([*existing["source_urls"].splitlines(), *person["source_urls"]]))
            if merged != existing["source_urls"]:
                existing["source_urls"] = merged
                changed = True
        else:
            records.append(row)
            known[key] = row
            changed = True
    if changed:
        df = pd.DataFrame(records, columns=EVENT_PEOPLE_COLUMNS)
        _write_table(df, path)
    return df


@transactional
def mark_event_researched(event_id, path=EVENTS_FILE):
    df = load_events(path)
    df.loc[df["event_id"].eq(event_id), "research_complete"] = "yes"
    _write_table(df, path)
    return df
