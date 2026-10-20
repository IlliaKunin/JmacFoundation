# JmacFoundation

Streamlit application for donor research and seller-side contacts from recent US business transactions.

## Run locally

Use Python 3.12 or newer, create a virtual environment, and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your Tavily and Groq keys in `.env`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Without `DATABASE_URL`, data stays in local CSV files. With `DATABASE_URL`, all users share persistent data in PostgreSQL: news, donors, source history, contact caches, manual edits and deletions. Database mode does not write back into the local CSV files or GitHub.

## Deploy on Streamlit Community Cloud

1. Push this project's source files to your GitHub repository. The `.gitignore` excludes keys, local data, contact lists and backups.
2. Create a persistent PostgreSQL database with your database provider. Obtain its PostgreSQL connection string (direct, session-pooler and transaction-pooler connections are supported). The application creates its `jmac_private.research_documents` table automatically; the database user must have schema/table-creation permission. The private schema keeps research data out of Supabase's public Data API. Automatic prepared statements are disabled for transaction poolers.
3. In Streamlit Community Cloud, create an app from your repository and select `app.py`. Use Python 3.12 or newer.
4. In **Advanced settings / Secrets**, add the top-level settings from `.streamlit/secrets.toml.example` with real keys and `DATABASE_URL`.
5. Deploy, then choose public or restricted access in the app's sharing settings.

For a step-by-step Supabase setup in Russian, see [DEPLOYMENT_RU.md](DEPLOYMENT_RU.md).

`DATABASE_URL` is required for persistent shared data in Community Cloud. Local CSV/SQLite files there are not durable storage. If the database is unavailable, the application raises an error instead of silently saving data locally. Connection details and API keys belong in Secrets, never in GitHub or chat.

All users see the same table on loading or refreshing the application. New results are appended, saved sources and contact identities are reused, and manual edits/deletions are shared. Database transactions serialize each read/modify/write operation so simultaneous saves do not overwrite other users' appended rows. Searches use the configured Tavily/Groq account limits.

Source code updates are deployed by pushing to GitHub; research results remain in the database across redeployments. Visitors can currently search, edit and delete shared records. If access should be limited, keep the deployment private until roles/authentication are added.

### Import existing CSV data once

On the computer holding your current data, set the **same** `DATABASE_URL` in `.env` and run:

```powershell
.\.venv\Scripts\python.exe manage_storage.py import
```

Import before other people start using the deployed app. It copies existing research/history/contact files into empty shared documents and never replaces an existing shared document. Your original local files are preserved. No CSV files need to be uploaded to GitHub.

### Export for Excel

With the same `DATABASE_URL` configured locally:

```powershell
.\.venv\Scripts\python.exe manage_storage.py export --output exports/sellers.csv
```

The UTF-8 CSV contains the combined seller/contact table and can be opened in Excel. This is a snapshot; run the command again for a fresh export.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -q
```

Tests mock paid APIs and use temporary files/databases. Shared-storage tests exercise concurrent inserts, duplicate protection, rollback, imports and edits with SQLite. PostgreSQL adapter behavior is tested separately with a mocked connection; an actual PostgreSQL connection must also be verified during deployment.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [data connections and persistence](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data).
