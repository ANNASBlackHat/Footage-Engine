"""Mint a Google Drive user OAuth token (one-time consent flow).

Why this exists: a service account has ``storageQuota.limit == 0`` and Drive
rejects its uploads to an ordinary folder with ``403 Service Accounts do not
have storage quota``. Uploads that run as a consenting *user* are owned by that
user and consume their quota, so this is the only way to write footage into a
normal personal Drive folder.

Setup (once, in Google Cloud Console):
    1. APIs & Services > Credentials > Create credentials > OAuth client ID
    2. Application type: **Desktop app** (a redirect URI cannot be kept secret,
       so a client secret on a public client is not a real secret anyway)
    3. Download the client JSON; pass it with --client-secrets

Then:
    pip install "footage-engine[gdrive]" google-auth-oauthlib
    python scripts/gdrive_oauth_login.py \
        --client-secrets ~/Downloads/client_secret_*.json \
        --out ~/.config/footage-engine/gdrive-oauth-token.json

The flow opens a browser, you sign in as the Drive account that should own the
uploads, and the token lands at --out. Point GDRIVE_OAUTH_CREDENTIALS_FILE at
that path.

Note on token lifetime: while the OAuth consent screen is in **Testing**
publishing status Google expires refresh tokens after 7 days, which silently
breaks a long-running worker. Set the consent screen to **Production** to get a
long-lived token; an unverified app then shows a warning screen you click past
once per consent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The full Drive scope, matching the backend's default. A narrower scope cannot
# read a folder the app did not create, which breaks exists()/get_local_path().
SCOPES = ["https://www.googleapis.com/auth/drive"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--client-secrets",
        required=True,
        help="Path to the downloaded OAuth client JSON (Desktop app).",
    )
    parser.add_argument(
        "--out",
        default="~/.config/footage-engine/gdrive-oauth-token.json",
        help="Where to write the token JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--scopes",
        default=",".join(SCOPES),
        help="Comma-separated scopes. Must match the scopes the backend requests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    secrets_path = Path(args.client_secrets).expanduser()
    if not secrets_path.is_file():
        print(f"error: client secrets not found: {secrets_path}", file=sys.stderr)
        return 1

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            'error: missing dependency. Install with: pip install google-auth-oauthlib',
            file=sys.stderr,
        )
        return 1

    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes=scopes)

    # run_local_server opens a browser and serves the redirect on localhost,
    # which is what a Desktop-app client is allowed to do.
    creds = flow.run_local_server(port=0)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The same shape parse_oauth_credentials() validates, and the same shape
    # `gcloud auth application-default login` writes.
    payload = {
        "type": "authorized_user",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri or "https://oauth2.googleapis.com/token",
        "scopes": list(creds.scopes or scopes),
    }
    if not payload["refresh_token"]:
        print(
            "error: no refresh_token in the response — the app is still in Testing "
            "publishing status, or the consent was for a different scope set.",
            file=sys.stderr,
        )
        return 1

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_path.chmod(0o600)

    identity = (creds.id_token or {}).get("email", "(id_token not requested)")
    print(f"Signed in as: {identity}")
    print(f"Token written to: {out_path}")
    print()
    print("Add to .env:")
    print("  STORAGE_BACKEND=gdrive")
    print(f"  GDRIVE_OAUTH_CREDENTIALS_FILE={out_path}")
    print("  GDRIVE_FOLDER_ID=<a folder that account can write to>")
    return 0


if __name__ == "__main__":
    sys.exit(main())