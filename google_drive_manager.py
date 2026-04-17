import json
import os
from pathlib import Path
from io import BytesIO
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

try:
    import streamlit as st
except Exception:
    st = None

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_FILE = Path("google_service_account.json")
FOLDER_ID = "1EhNX1z3bwKGuwAdZVoDvvrn2-XqFcBF2"                              


def _get_config_value(key: str, default=None):
    value = os.getenv(key)
    if value not in (None, ""):
        return value

    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    return default


def _get_query_param(key: str):
    if st is None:
        return None

    try:
        value = st.query_params.get(key)
    except Exception:
        try:
            value = st.experimental_get_query_params().get(key)
        except Exception:
            return None

    if isinstance(value, list):
        return value[0] if value else None
    return value


def _load_client_config() -> dict | None:
    client_payload = _get_config_value("GOOGLE_OAUTH_CLIENT_JSON")
    if not client_payload:
        return None

    if isinstance(client_payload, dict):
        return dict(client_payload)

    if isinstance(client_payload, str):
        return json.loads(client_payload)

    raise ValueError("GOOGLE_OAUTH_CLIENT_JSON must be a JSON string or mapping.")


def _load_service_account_config() -> dict | None:
    if st is not None:
        # Common Streamlit pattern: [gcp_service_account] table in secrets.toml
        try:
            gcp_service_account = st.secrets.get("gcp_service_account")
            if isinstance(gcp_service_account, dict):
                return dict(gcp_service_account)
        except Exception:
            pass

        # Alternate table names often used in projects
        try:
            service_account_section = st.secrets.get("service_account")
            if isinstance(service_account_section, dict):
                return dict(service_account_section)
        except Exception:
            pass

    service_account_payload = _get_config_value("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_payload:
        return None

    if isinstance(service_account_payload, dict):
        return dict(service_account_payload)

    if isinstance(service_account_payload, str):
        return json.loads(service_account_payload)

    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON must be a JSON string or mapping.")


def _extract_redirect_uri(client_config: dict | None) -> str | None:
    if not client_config:
        return None

    for section_name in ("web", "installed"):
        section = client_config.get(section_name) or {}
        redirect_uris = section.get("redirect_uris") or []
        if redirect_uris:
            return redirect_uris[0]

    return None


def _get_streamlit_app_url() -> str:
    """
    Detect the Streamlit app URL from environment or st.
    For Streamlit Cloud, this should be https://yourapp.streamlit.app/
    """
    # Try environment variable first
    app_url = os.getenv("STREAMLIT_SERVER_BASE_URL_PATH")
    if app_url:
        return app_url.rstrip("/") + "/"
    
    # Try Streamlit secrets
    if st is not None:
        try:
            app_url = st.secrets.get("STREAMLIT_APP_URL")
            if app_url:
                return app_url.rstrip("/") + "/"
        except Exception:
            pass
    
    # Default for Streamlit Cloud (user must set this)
    return "https://aarogyaveda.streamlit.app/"


class GoogleDriveManager:
    def __init__(self):
        """Initialize Google Drive Manager with service-account-first authentication."""
        self.creds = None
        self.folder_id = _get_config_value("GOOGLE_DRIVE_FOLDER_ID", FOLDER_ID)
        self.auth_url = None
        self.is_authenticated = False
        self.auth_mode = None
        self.last_error = None
        self._authenticate()
        if self.creds:
            try:
                self.service = build("drive", "v3", credentials=self.creds)
                self.is_authenticated = True
            except Exception as e:
                self.service = None
                self.is_authenticated = False
        else:
            self.service = None

    def _is_oauth_enabled(self) -> bool:
        raw = str(_get_config_value("ENABLE_DRIVE_OAUTH", "false")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _authenticate_with_refresh_token(self) -> bool:
        client_id = _get_config_value("GOOGLE_CLIENT_ID")
        client_secret = _get_config_value("GOOGLE_CLIENT_SECRET")
        refresh_token = _get_config_value("GOOGLE_REFRESH_TOKEN")

        if not (client_id and client_secret and refresh_token):
            return False

        try:
            self.creds = Credentials(
                token=None,
                refresh_token=str(refresh_token),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=str(client_id),
                client_secret=str(client_secret),
                scopes=SCOPES,
            )
            self.creds.refresh(Request())
            self.auth_mode = "personal-oauth"
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = f"Invalid refresh-token configuration: {str(e)}"
            return False

    def _authenticate_with_oauth(self) -> bool:
        if not self._is_oauth_enabled():
            return False

        token_json = None
        if st is not None:
            token_json = st.session_state.get("_drive_token_json")

        if token_json:
            try:
                self.creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
                if self.creds.valid:
                    self.auth_mode = "oauth"
                    return True
                if self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    if st is not None:
                        st.session_state["_drive_token_json"] = self.creds.to_json()
                    self.auth_mode = "oauth"
                    return True
            except Exception:
                self.creds = None

    def _authenticate(self):
        # Primary strategy: personal OAuth via refresh token (no browser popup on cloud)
        if self._authenticate_with_refresh_token():
            self.auth_url = None
            return

        # Optional strategy: browser OAuth fallback (only if enabled)
        client_config = _load_client_config()
        if client_config and self._is_oauth_enabled():
            try:
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

                redirect_uri = (
                    _get_config_value("GOOGLE_OAUTH_REDIRECT_URI")
                    or _get_streamlit_app_url()
                    or _extract_redirect_uri(client_config)
                )

                if redirect_uri:
                    flow.redirect_uri = redirect_uri

                if self._authenticate_with_oauth():
                    return

                auth_code = _get_query_param("code")
                if auth_code:
                    try:
                        flow.fetch_token(code=auth_code)
                        self.creds = flow.credentials

                        if st is not None:
                            st.session_state["_drive_token_json"] = self.creds.to_json()
                            st.session_state["_oauth_authenticated"] = True

                        self.auth_mode = "oauth"
                        self.auth_url = None
                        return
                    except Exception:
                        self.creds = None

                self.auth_url, _ = flow.authorization_url(
                    access_type="offline",
                    include_granted_scopes="true",
                    prompt="consent",
                )
                self.last_error = None
                return
            except Exception:
                self.creds = None
                self.last_error = "OAuth initialization failed."

        if not self.last_error:
            self.last_error = (
                "Missing Google Drive credentials. Add GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN in Streamlit secrets."
            )

        raise FileNotFoundError(
            "Google Drive credentials not configured. "
            "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN in Streamlit secrets. "
            "If you still want browser OAuth, also set ENABLE_DRIVE_OAUTH=true."
        )
    
    def upload_pdf(self, pdf_buffer: BytesIO, filename: str) -> dict:
        """
        Upload PDF to Google Drive folder
        
        Args:
            pdf_buffer: BytesIO object containing PDF data
            filename: Name for the file in Drive
            
        Returns:
            dict with file_id, filename, and download_url
        """
        try:
            if not self.service:
                return {"success": False, "error": "Google Drive is not authenticated."}

            file_metadata = {"name": filename, "mimeType": "application/pdf"}
            if self.folder_id:
                file_metadata["parents"] = [self.folder_id]

            pdf_buffer.seek(0)

            media = MediaIoBaseUpload(
                pdf_buffer,
                mimetype="application/pdf",
                resumable=True
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, webContentLink"
            ).execute()

            download_url = f"https://drive.google.com/uc?export=download&id={file['id']}"
            view_url = file.get("webViewLink") or f"https://drive.google.com/file/d/{file['id']}/view"
            
            return {
                "success": True,
                "file_id": file["id"],
                "filename": file["name"],
                "download_url": download_url,
                "view_url": view_url,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def upload_file(self, file_buffer: BytesIO, filename: str, mime_type: str, replace_existing: bool = True) -> dict:
        """Upload or replace a generic file in Google Drive."""
        try:
            if not self.service:
                return {"success": False, "error": "Google Drive is not authenticated."}

            file_metadata = {"name": filename, "mimeType": mime_type}
            if self.folder_id:
                file_metadata["parents"] = [self.folder_id]

            file_buffer.seek(0)
            media = MediaIoBaseUpload(file_buffer, mimetype=mime_type, resumable=True)

            existing_file = None
            if replace_existing:
                existing_file = self.get_file_by_name(filename)

            if existing_file:
                file = self.service.files().update(
                    fileId=existing_file["id"],
                    body={"name": filename},
                    media_body=media,
                    fields="id, name, webViewLink, webContentLink",
                ).execute()
            else:
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, webViewLink, webContentLink",
                ).execute()

            download_url = f"https://drive.google.com/uc?export=download&id={file['id']}"
            view_url = file.get("webViewLink") or f"https://drive.google.com/file/d/{file['id']}/view"
            return {
                "success": True,
                "file_id": file["id"],
                "filename": file["name"],
                "download_url": download_url,
                "view_url": view_url,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_file_by_name(self, filename: str) -> dict | None:
        """Get the most recently modified file with the given name from Drive."""
        try:
            if not self.service:
                return None

            query = f"name='{filename}' and trashed=false"
            if self.folder_id:
                query = f"'{self.folder_id}' in parents and {query}"

            results = self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id, name, createdTime, modifiedTime, webViewLink)",
                pageSize=10,
                orderBy="modifiedTime desc",
            ).execute()

            files = results.get("files", [])
            return files[0] if files else None
        except Exception:
            return None

    def download_file_bytes(self, file_id: str) -> bytes | None:
        """Download a file from Drive as raw bytes."""
        try:
            if not self.service:
                return None

            request = self.service.files().get_media(fileId=file_id)
            stream = BytesIO()
            downloader = MediaIoBaseDownload(stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            return stream.getvalue()
        except Exception:
            return None

    def download_file_by_name(self, filename: str) -> bytes | None:
        """Download the most recent file with the given name from Drive."""
        file_info = self.get_file_by_name(filename)
        if not file_info:
            return None
        return self.download_file_bytes(file_info["id"])
    
    def get_file_list(self, limit: int = 20) -> list:
        """Get list of recent reports from Drive folder"""
        try:
            query = "mimeType='application/pdf' and trashed=false"
            if self.folder_id:
                query = f"'{self.folder_id}' in parents and {query}"
            results = self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id, name, createdTime, size, webViewLink)",
                pageSize=limit,
                orderBy="createdTime desc"
            ).execute()
            
            files = results.get("files", [])
            
                                             
            for file in files:
                file["download_url"] = f"https://drive.google.com/uc?export=download&id={file['id']}"
                file["view_url"] = file.get("webViewLink") or f"https://drive.google.com/file/d/{file['id']}/view"
            
            return files
        
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Drive"""
        try:
            self.service.files().delete(fileId=file_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

