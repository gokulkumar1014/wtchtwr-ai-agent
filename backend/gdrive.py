from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseUpload
except ImportError:  # pragma: no cover - optional dependency
    service_account = None  # type: ignore
    build = None  # type: ignore
    HttpError = Exception  # type: ignore
    MediaIoBaseUpload = None  # type: ignore

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DriveConfig:
    service_account_file: Optional[str]
    service_account_json: Optional[str]
    folder_id: Optional[str]
    share_with_anyone: bool = True


class DriveUploader:
    def __init__(self) -> None:
        self._config = DriveConfig(
            service_account_file=os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE"),
            service_account_json=os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON"),
            folder_id=os.getenv("GDRIVE_UPLOAD_FOLDER_ID"),
            share_with_anyone=_bool_env("GDRIVE_SHARE_WITH_ANYONE", True),
        )
        self._service = None
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(
            (self._config.service_account_file or self._config.service_account_json)
            and self._config.folder_id
            and build is not None
            and service_account is not None
            and MediaIoBaseUpload is not None
        )

    def _get_service(self):
        if not self.enabled():
            raise RuntimeError(
                "Google Drive upload is not configured. Set GDRIVE_SERVICE_ACCOUNT_FILE (or GDRIVE_SERVICE_ACCOUNT_JSON) "
                "and GDRIVE_UPLOAD_FOLDER_ID, then install google-api-python-client."
            )
        with self._lock:
            if self._service is not None:
                return self._service
            if self._config.service_account_json:
                creds = service_account.Credentials.from_service_account_info(
                    json.loads(self._config.service_account_json), scopes=_DEFAULT_SCOPES
                )
            else:
                creds = service_account.Credentials.from_service_account_file(
                    self._config.service_account_file, scopes=_DEFAULT_SCOPES
                )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            return self._service

    def upload_bytes(
        self,
        *,
        name: str,
        content: bytes,
        mimetype: str,
        share_with: Optional[str] = None,
    ) -> str:
        service = self._get_service()
        metadata = {"name": name}
        if self._config.folder_id:
            metadata["parents"] = [self._config.folder_id]
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mimetype, resumable=False)
        try:
            file = (
                service.files()
                .create(body=metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )
        except HttpError as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Unable to upload file to Google Drive: {exc}") from exc
        file_id = file.get("id")
        if not file_id:
            raise RuntimeError("Google Drive upload did not return a file id.")

        if self._config.share_with_anyone:
            permission_body = {"type": "anyone", "role": "reader"}
            service.permissions().create(fileId=file_id, body=permission_body).execute()
        elif share_with:
            permission_body = {"type": "user", "role": "reader", "emailAddress": share_with}
            service.permissions().create(
                fileId=file_id, body=permission_body, sendNotificationEmail=False
            ).execute()

        link = file.get("webViewLink")
        if not link:
            raise RuntimeError("Google Drive upload did not return a shareable link.")
        return link


drive_uploader = DriveUploader()