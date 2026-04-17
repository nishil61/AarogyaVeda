from __future__ import annotations

from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CLIENT_SECRET_PATH = Path("client_secret.json")


def main() -> None:
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError("client_secret.json not found in project root.")

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(
        host="localhost",
        port=0,
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )

    # `flow.client_config` can be either nested (`{"installed": {...}}`) or already flattened.
    cfg = flow.client_config
    client_config = cfg.get("installed") or cfg.get("web") or cfg
    client_id = client_config.get("client_id", "")
    client_secret = client_config.get("client_secret", "")

    print("\n=== COPY THESE INTO STREAMLIT SECRETS ===")
    print(f"GOOGLE_CLIENT_ID={client_id}")
    print(f"GOOGLE_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
    print("========================================\n")


if __name__ == "__main__":
    main()
