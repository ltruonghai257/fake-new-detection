import io
import os
import mimetypes
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import pickle

_SCOPES = ["https://www.googleapis.com/auth/drive"]

_MIME_MAP = {
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    # trained model files
    ".h5": "application/octet-stream",
    ".hdf5": "application/octet-stream",
    ".pkl": "application/octet-stream",
    ".pickle": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".pth": "application/octet-stream",
    ".safetensors": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".ckpt": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".zip": "application/zip",
}


class GoogleDriveUploader:
    """Upload files to Google Drive using a service account or OAuth credentials.

    Authentication options
    ----------------------
    Service account (recommended for automation):
        uploader = GoogleDriveUploader(credentials_path="service-account.json")

    OAuth (for personal/interactive use):
        uploader = GoogleDriveUploader(credentials_path="oauth-client.json", use_oauth=True)
        # On the first run a browser window opens to authorise access.
        # A token cache is saved next to credentials_path as 'token.pickle'.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        use_oauth: bool = False,
    ) -> None:
        if credentials_path is None:
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError(
                "Provide credentials_path or set GOOGLE_APPLICATION_CREDENTIALS env var."
            )

        self._creds_path = Path(credentials_path)
        self._use_oauth = use_oauth
        self._service = build("drive", "v3", credentials=self._authenticate())

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _authenticate(self):
        if self._use_oauth:
            return self._oauth_flow()
        return service_account.Credentials.from_service_account_file(
            str(self._creds_path), scopes=_SCOPES
        )

    def _oauth_flow(self) -> Credentials:
        token_path = self._creds_path.parent / "token.pickle"
        creds = None
        if token_path.exists():
            with open(token_path, "rb") as f:
                creds = pickle.load(f)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._creds_path), _SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token_path, "wb") as f:
                pickle.dump(creds, f)
        return creds

    # ------------------------------------------------------------------
    # Folder helpers
    # ------------------------------------------------------------------

    def _get_or_create_folder(
        self, folder_name: str, parent_id: Optional[str] = None
    ) -> str:
        """Return the Drive folder ID, creating it if it does not exist."""
        query = (
            f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            " and trashed=false"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = (
            self._service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        items = results.get("files", [])
        if items:
            return items[0]["id"]

        meta = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            meta["parents"] = [parent_id]
        folder = self._service.files().create(body=meta, fields="id").execute()
        return folder["id"]

    def _resolve_folder_path(self, folder_path: str) -> str:
        """Accept 'a/b/c' and recursively create/get each segment."""
        parts = [p for p in folder_path.replace("\\", "/").split("/") if p]
        parent_id = None
        for part in parts:
            parent_id = self._get_or_create_folder(part, parent_id)
        return parent_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        folder_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Upload a single file to Drive.

        Parameters
        ----------
        local_path:
            Absolute or relative path to the local file.
        folder_path:
            Drive folder name or nested path (e.g. ``"project/data"``).
            The folder is created if it does not exist.
            If *None*, the file is placed at the root of the Drive.
        overwrite:
            If *True*, an existing file with the same name in the same folder
            is deleted before uploading.

        Returns
        -------
        str
            The Google Drive file ID of the uploaded file.
        """
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {local_path}")

        mime_type = _MIME_MAP.get(path.suffix.lower()) or (
            mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        )

        parent_id: Optional[str] = None
        if folder_path:
            parent_id = self._resolve_folder_path(folder_path)

        if overwrite and parent_id:
            self._delete_if_exists(path.name, parent_id)

        meta = {"name": path.name}
        if parent_id:
            meta["parents"] = [parent_id]

        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        file = (
            self._service.files()
            .create(body=meta, media_body=media, fields="id")
            .execute()
        )
        file_id = file.get("id")
        print(f"Uploaded '{path.name}' → Drive file ID: {file_id}")
        return file_id

    def upload_directory(
        self,
        local_dir: str,
        folder_path: Optional[str] = None,
        extensions: Optional[list] = None,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> list[str]:
        """Upload all matching files from a local directory.

        Parameters
        ----------
        local_dir:
            Local directory to scan for files.
        folder_path:
            Target Drive folder path (created if missing).
        extensions:
            Whitelist of extensions to upload, e.g. ``['.json', '.png']``.
            Defaults to all supported types.
        recursive:
            If *True*, also upload files inside sub-directories (maintaining
            the folder structure on Drive).
        overwrite:
            Delete existing Drive file before re-uploading.

        Returns
        -------
        list[str]
            List of Drive file IDs that were uploaded.
        """
        dir_path = Path(local_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {local_dir}")

        allowed = set(extensions) if extensions else set(_MIME_MAP)
        pattern = "**/*" if recursive else "*"
        uploaded_ids: list[str] = []

        for file in dir_path.glob(pattern):
            if not file.is_file():
                continue
            if file.suffix.lower() not in allowed:
                continue

            if recursive and folder_path:
                relative = file.parent.relative_to(dir_path)
                target = (
                    f"{folder_path}/{relative}" if str(relative) != "." else folder_path
                )
            else:
                target = folder_path

            file_id = self.upload_file(str(file), folder_path=target, overwrite=overwrite)
            uploaded_ids.append(file_id)

        print(f"Uploaded {len(uploaded_ids)} file(s) from '{local_dir}'.")
        return uploaded_ids

    def _delete_if_exists(self, file_name: str, parent_id: str) -> None:
        query = (
            f"name='{file_name}' and '{parent_id}' in parents and trashed=false"
        )
        results = (
            self._service.files()
            .list(q=query, spaces="drive", fields="files(id)")
            .execute()
        )
        for item in results.get("files", []):
            self._service.files().delete(fileId=item["id"]).execute()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _find_folder_id(self, folder_path: str) -> Optional[str]:
        """Resolve a nested folder path to its Drive ID. Returns None if not found."""
        parts = [p for p in folder_path.replace("\\", "/").split("/") if p]
        parent_id = None
        for part in parts:
            query = (
                f"name='{part}' and mimeType='application/vnd.google-apps.folder'"
                " and trashed=false"
            )
            if parent_id:
                query += f" and '{parent_id}' in parents"
            results = (
                self._service.files()
                .list(q=query, spaces="drive", fields="files(id)")
                .execute()
            )
            items = results.get("files", [])
            if not items:
                return None
            parent_id = items[0]["id"]
        return parent_id

    def download_file(self, file_id: str, local_path: str) -> None:
        """Download a single Drive file by its file ID.

        Parameters
        ----------
        file_id:
            Google Drive file ID.
        local_path:
            Destination path on the local machine (including file name).
        """
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        request = self._service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        dest.write_bytes(buffer.getvalue())
        print(f"Downloaded → {local_path}")

    def download_folder(
        self,
        folder_path: str,
        local_dir: str,
        extensions: Optional[list] = None,
        recursive: bool = False,
    ) -> list[str]:
        """Download all matching files from a Drive folder to a local directory.

        Parameters
        ----------
        folder_path:
            Drive folder name or nested path (e.g. ``"project/data"``).
        local_dir:
            Local destination directory (created if it does not exist).
        extensions:
            Whitelist of extensions, e.g. ``['.json', '.png']``.
            If *None*, all files in the folder are downloaded.
        recursive:
            If *True*, also download files inside sub-folders, mirroring the
            folder structure locally.

        Returns
        -------
        list[str]
            List of local paths that were written.
        """
        folder_id = self._find_folder_id(folder_path)
        if not folder_id:
            raise FileNotFoundError(f"Drive folder not found: '{folder_path}'")

        local_root = Path(local_dir)
        local_root.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        self._download_folder_recursive(
            folder_id, local_root, extensions, recursive, downloaded
        )
        print(f"Downloaded {len(downloaded)} file(s) to '{local_dir}'.")
        return downloaded

    def _download_folder_recursive(
        self,
        folder_id: str,
        local_dir: Path,
        extensions: Optional[list],
        recursive: bool,
        downloaded: list,
    ) -> None:
        page_token = None
        while True:
            query = f"'{folder_id}' in parents and trashed=false"
            response = (
                self._service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )
            for item in response.get("files", []):
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    if recursive:
                        sub_dir = local_dir / item["name"]
                        sub_dir.mkdir(parents=True, exist_ok=True)
                        self._download_folder_recursive(
                            item["id"], sub_dir, extensions, recursive, downloaded
                        )
                else:
                    ext = Path(item["name"]).suffix.lower()
                    if extensions and ext not in extensions:
                        continue
                    dest = str(local_dir / item["name"])
                    self.download_file(item["id"], dest)
                    downloaded.append(dest)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
