import logging
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from main import run_workflow
from news_workflow import run_news_workflow
from seller_contacts import seller_article_table
from storage import count_news_articles, save_news_table_edits, load_donors, load_funding_research, load_news_articles, load_seller_contacts, unique_donor_names


load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="Donor & News Search", page_icon="🔎", layout="wide")

st.title("Hello! Let's start your research.")


def show_research():
    st.write(
        "We take donors found in news articles and other sources, then research what "
        "they fund, whether you can apply, and how to contact them. Choose how many new "
        "articles or websites to read. We research every new donor found and any saved "
        "donors that have not been researched yet. Saved sources and researched donors are skipped."
    )
    match_column, applications_column = st.columns(2)
    with match_column:
        st.markdown(
            "**Funding Match**\n\n"
            "- **Yes** — funds at least one of our focus areas: education, AI, STEM, or youth programs.\n"
            "- **No** — its funding scope does not match our focus areas.\n"
            "- **Unknown** — not enough information to determine a match."
        )
    with applications_column:
        st.markdown(
            "**Applications**\n\n"
            "- **Open** — applications can currently be submitted without an invitation.\n"
            "- **Closed** — applications are closed or the published deadlines have passed.\n"
            "- **Invitation Only** — only applicants invited by the funder may apply.\n"
            "- **Unknown** — current application availability could not be established."
        )
    st.write(
        "**Name** — the donor or foundation being researched. "
        "**Contact Method** — an official application link, grants email, contact form, or other way to get in touch."
    )
    donors = load_donors()
    saved_names = unique_donor_names(donors)
    requested_count = st.number_input(
        "Number of new articles or websites", min_value=1, max_value=20,
        value=1, step=1, key="article_count",
        help="3 means three new readable sources, not three donors. We look for replacements when sources are unavailable or already saved.",
    )
    st.caption(
        f"{len(saved_names)} saved donors. "
        f"This run will look for {requested_count} new sources and research all unstudied donor names."
    )
    df = load_funding_research()
    if st.button("Search and research donors", type="primary", key="start_workflow"):
        with st.status("Finding new sources and researching donors...", expanded=True) as status:
            try:
                df = run_workflow(
                    article_count=int(requested_count),
                    on_progress=lambda message: status.update(label=message),
                )
            except Exception:
                logging.exception("Donor research failed")
                status.update(label="Research could not be completed", state="error", expanded=False)
                st.error("Research could not be completed. Any saved results are available below. Please try again.")
                df = load_funding_research()
            else:
                status.update(label="Research complete", state="complete", expanded=False)
                processed_count = df.attrs.get("processed_articles", 0)
                researched_count = df.attrs.get("researched_donors", 0)
                new_donors = df.attrs.get("new_donors", 0)
                if processed_count < requested_count:
                    st.warning(
                        f"Processed {processed_count} of {requested_count} requested new sources "
                        "after the additional searches. Available donor results have been saved."
                    )
                st.success(
                    f"Processed {processed_count} new sources, added {new_donors} donors, "
                    f"and researched {researched_count} donors."
                )

    st.subheader("Funding opportunities")
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={"Contact Method": st.column_config.TextColumn("Contact Method", width="large")},
    )


def show_news_search():
    if notice := st.session_state.pop("news_edit_notice", None):
        st.success(notice)
    st.write(
        "Discover people and companies connected to recent major financial events involving US businesses. "
        "We research news from the last 30 days about announced or completed business sales, acquisitions, "
        "stake sales and investor exits, then identify seller-side participants, their roles and available "
        "professional contact details. Choose how many new articles to research; each article may produce "
        "several people or organizations."
    )
    st.write(
        "Make the table your own: add or change text in editable cells, update contact details, "
        "or remove any row. Tick Delete for rows you want to remove, then click Save changes "
        "to save your edits and deletions."
    )
    st.caption("US companies and businesses | Business sales | Acquisitions | Secondary share sales | Investor exits | Last 30 days")
    max_results = st.number_input(
        "Number of new articles to save", min_value=1, max_value=20, value=1, step=1,
        key="news_max_results",
        help="3 means three qualifying articles and all their seller-side participants, with automatic contact research. Excluded articles do not count. Interrupted articles resume first. Search stops after three consecutive rounds without new sources.",
    )
    df = load_news_articles()
    if st.button("Find articles and seller contacts", type="primary", key="start_news_search"):
        with st.status("Finding articles, sellers and their contacts...", expanded=True) as status:
            try:
                df = run_news_workflow(
                    article_count=int(max_results),
                    on_progress=lambda message: status.update(label=message),
                )
            except Exception:
                logging.exception("News search failed")
                status.update(label="News search could not be completed", state="error", expanded=False)
                st.error("Research could not be completed. Saved results are below. Start again to resume interrupted articles and skip completed contact checks.")
                df = load_news_articles()
            else:
                status.update(label="News search complete", state="complete", expanded=False)
                processed = df.attrs.get("processed_articles", 0)
                saved = df.attrs.get("saved_articles", 0)
                excluded = df.attrs.get("excluded_articles", 0)
                st.success(
                    f"Checked {processed} new articles: saved {saved}, excluded {excluded}. "
                    f"Researched {df.attrs.get('checked_sellers', 0)} new seller identities. "
                    f"{count_news_articles(df)} articles saved in total."
                )
                if saved < max_results:
                    st.warning(
                        f"Saved {saved} of {max_results} requested new articles. "
                        "No new sources were found after repeated searches. "
                        "All qualifying results have been saved."
                    )
    df = seller_article_table(df, load_seller_contacts())
    st.caption(
        f"{count_news_articles(df)} saved articles | "
        f"{len(df)} seller entries. One row per seller-side participant per article. "
        "All saved results, including this run."
    )
    st.subheader("Sellers and contacts")
    st.caption("Edit a cell directly. To remove rows, tick Delete. Click Save changes to save your edits and deletions to CSV.")
    # Refresh widget state whenever saved data changes, so old edits cannot
    # silently apply to different records after a search or save.
    fingerprint = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
    editable = df.copy()
    editable.insert(0, "_delete", False)
    edited = st.data_editor(
        editable, width="stretch", hide_index=True, num_rows="fixed",
        key=f"news_table_{fingerprint}",
        column_order=[
            "_delete", "participant_name", "participant_type", "role",
            "email", "phone", "contact_page", "linkedin",
            "url", "published_date",
        ],
        column_config={
            "_delete": st.column_config.CheckboxColumn("Delete", default=False),
            "article_title": None,
            "published_date": st.column_config.TextColumn("Publication date"),
            "description": None,
            "participant_name": st.column_config.TextColumn("Person / organization", width="large"),
            "participant_type": st.column_config.TextColumn("Type"),
            "organization": None,
            "_row_revision": None,
            "role": st.column_config.TextColumn("Role", width="large"),
            "deal_side": None,
            "url": st.column_config.LinkColumn("Article URL"),
            "email": st.column_config.TextColumn("Email"),
            "phone": st.column_config.TextColumn("Phone"),
            "contact_page": st.column_config.LinkColumn("Contact page"),
            "linkedin": st.column_config.LinkColumn("LinkedIn"),
            "contact_sources": None,
        },
    )
    if st.button("Save changes", key="save_news_edits", disabled=df.empty):
        try:
            result = save_news_table_edits(df, edited)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            logging.exception("Could not save news table edits")
            st.error("Could not save your changes. Please try again.")
        else:
            deleted = result.attrs.get("deleted_rows", 0)
            updated = result.attrs.get("updated_rows", 0)
            st.session_state["news_edit_notice"] = f"Saved changes to {updated} rows. Deleted {deleted} rows."
            st.rerun()

donor_tab, news_tab = st.tabs(["Donor Research", "News Search"])
with donor_tab:
    show_research()
with news_tab:
    show_news_search()
