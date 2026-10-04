# Kaggle kernels

Notebooks that run on Kaggle's GPU infrastructure, pushed with the `kaggle` CLI
instead of being clicked through the web UI.

## `yt_ingest/` — YouTube ingest

Clones the repo, rebuilds `.env` from Kaggle secrets, pulls YouTube cookies from
your worker endpoint, and ingests each URL into scene chunks.

### One-time setup

1. Install and authenticate the CLI. Kaggle CLI 2.x no longer uses the old
   `kaggle.json` username/key pair:

   ```bash
   pip install kaggle
   # https://www.kaggle.com/settings/api -> "Generate New Token",
   # then either:
   export KAGGLE_API_TOKEN=<token>          # option A
   # or save the token to ~/.kaggle/access_token   (option B)
   kaggle kernels status annasblackhat/footage-engine-yt-ingest
   ```

2. Create the kernel once in the UI so secrets can be attached — the CLI cannot
   attach secrets, it can only push code:

   <https://www.kaggle.com/code/annasblackhat/footage-engine-yt-ingest>

   Or push first and attach secrets afterwards (the first run will fail on the
   missing-secret check, which is intentional).

3. Attach secrets under **Add-ons → Secrets**:

   | Secret | Required | Notes |
   |---|---|---|
   | `DATABASE_URL` | yes | Shared Postgres. **Not SQLite** — the session's local DB file is discarded when the kernel ends, so nothing is retrievable. |
   | `VECTOR_STORE` | yes | `zilliz`. With `in_memory` the kernel indexes into a private store and searches find nothing. |
   | `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `ZILLIZ_COLLECTION_NAME` | yes | X-CLIP collection (512d). |
   | `INGEST_URLS` | yes | Comma-separated YouTube URLs. |
   | `YT_COOKIES_URL` | yes | Your `workers.dev` endpoint. Rotating it means updating this secret. |
   | `QWEN_ZILLIZ_URI`, `QWEN_ZILLIZ_TOKEN`, `QWEN_ZILLIZ_COLLECTION_NAME` | if `EMBEDDING_BACKEND=qwen` | Qwen is 2048d and cannot share the X-CLIP collection. |
   | `STORAGE_BACKEND` | if uploading | `gdrive`. |
   | `GDRIVE_OAUTH_CREDENTIALS_JSON` | if uploading | The whole `authorized_user` token, minified to one line. |
   | `GDRIVE_FOLDER_ID` | if uploading | Destination folder. |
   | `UPLOAD_RAW_TO_STORAGE`, `UPLOAD_CHUNKS_TO_STORAGE` | if uploading | `true` to upload. |
   | `INGEST_ENTITY`, `INGEST_ENTITY_TYPE` | no | Entity to tag the footage with. |

### Push and monitor

```bash
kaggle kernels push -p kaggle/yt_ingest
kaggle kernels status annasblackhat/footage-engine-yt-ingest
kaggle kernels output annasblackhat/footage-engine-yt-ingest -p ./kaggle_out
```

`kaggle kernels pull -p ./kaggle/yt_ingest` fetches the notebook back, including
any version Kaggle committed to it.

## Constraints worth knowing

- **No browser on Kaggle.** `YOUTUBE_COOKIES_FROM_BROWSER=chrome` cannot work
  here, which is why the notebook fetches `cookies.txt` over HTTPS and sets
  `YOUTUBE_COOKIES` to the path.
- **Secrets are readable by anyone who can open the kernel.** `kernel-metadata.json`
  sets `is_private: true`; keep it that way, or grant access only to specific
  people under the kernel's Share settings.
- **Never commit a cookie URL or key.** `kernel-metadata.json` and the notebook
  are safe in git precisely because every value arrives via `kaggle_secrets`.
- **Local SQLite is useless here** even when the ingest succeeds — the file dies
  with the session. Use a shared Postgres `DATABASE_URL` so results survive and a
  laptop worker can find them.