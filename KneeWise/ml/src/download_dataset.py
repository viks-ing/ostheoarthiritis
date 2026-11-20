#!/usr/bin/env python3
"""
download_dataset.py - KneeWise Mendeley Data API Downloader

Automates retrieval of the Knee Osteoarthritis Severity Grading Dataset
(56rmx5bjcr, v1) from Mendeley Data API.

Features:
- OAuth2/OIDC token authentication and client_credentials flow.
- Anonymous / Public API fallback for public metadata.
- Streaming chunked download with tqdm progress bar.
- Resumable downloads via HTTP Range headers.
- Network failure retry with exponential backoff.
- SHA-256 and file size integrity verification.
- Non-destructive archive validation (zipfile testzip).
- Dedicated --metadata-only and --verify-only CLI modes.
"""

import os
import sys
import argparse
import time
import json
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR

API_BASE = "https://api.data.mendeley.com"
PUBLIC_API_BASE = "https://data.mendeley.com/public-api"
AUTH_TOKEN_URL = "https://auth.data.mendeley.com/oauth2/token"

DEFAULT_DATASET_ID = "56rmx5bjcr"
DEFAULT_VERSION = 1
DEFAULT_ARCHIVE_NAME = "KneeXrayData.zip"
EXPECTED_APPROX_SIZE_GB = 6.68
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk


class MendeleyDownloader:
    """Manages authentication, metadata inspection, and resilient downloading."""

    def __init__(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        version: int = DEFAULT_VERSION,
        output_dir: Path = RAW_DATA_DIR,
        archive_name: str = DEFAULT_ARCHIVE_NAME,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.dataset_id = dataset_id
        self.version = version
        self.output_dir = Path(output_dir).resolve()
        self.archive_name = archive_name
        self.destination_path = self.output_dir / self.archive_name

        # Resolve credentials from arguments or environment variables
        self.access_token = (
            access_token
            or os.environ.get("MENDELEY_ACCESS_TOKEN")
            or os.environ.get("MENDELEY_TOKEN")
            or os.environ.get("MENDELEY_API_KEY")
        )
        self.client_id = client_id or os.environ.get("MENDELEY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("MENDELEY_CLIENT_SECRET")

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KneeWise-ML-Downloader/1.0 (Medical Image Research)",
        })

    def authenticate(self) -> bool:
        """
        Authenticates via Client Credentials flow if client_id and client_secret
        are provided and no direct access token is available.
        """
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"
            return True

        if self.client_id and self.client_secret:
            print("[*] Exchanging Client Credentials for OAuth2 Bearer token...")
            try:
                resp = requests.post(
                    AUTH_TOKEN_URL,
                    auth=(self.client_id, self.client_secret),
                    data={"grant_type": "client_credentials", "scope": "all"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    token_data = resp.json()
                    self.access_token = token_data.get("access_token")
                    self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                    print(f"[+] Successfully authenticated via OAuth2 (Token expires in: {token_data.get('expires_in')}s)")
                    return True
                else:
                    print(f"[!] OAuth2 authentication failed (HTTP {resp.status_code}): {resp.text}")
                    return False
            except Exception as e:
                print(f"[!] Error contacting authentication server: {e}")
                return False

        # Anonymous fallback
        return True

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """
        Retrieves dataset metadata:
        Primary: GET https://api.data.mendeley.com/datasets/{id}?version={version}
        Fallback: GET https://data.mendeley.com/public-api/datasets/{id}?version={version}
        """
        api_url = f"{API_BASE}/datasets/{self.dataset_id}?version={self.version}"
        print(f"[*] Querying API: {api_url}")

        resp = self.session.get(api_url, timeout=15)
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 401:
            print("[!] Endpoint returned 401 Unauthorized (OAuth Bearer token required for api.data.mendeley.com).")
            print("[*] Attempting fallback to public metadata endpoint...")
            pub_url = f"{PUBLIC_API_BASE}/datasets/{self.dataset_id}?version={self.version}"
            pub_resp = requests.get(pub_url, headers=self.session.headers, timeout=15)
            if pub_resp.status_code == 200:
                print("[+] Successfully retrieved public dataset metadata.")
                return pub_resp.json()

        resp.raise_for_status()
        return {}

    def get_zip_metadata(self) -> Dict[str, Any]:
        """
        Retrieves ZIP archive metadata:
        GET https://api.data.mendeley.com/datasets/{id}/zip?version={version}
        """
        zip_url = f"{API_BASE}/datasets/{self.dataset_id}/zip?version={self.version}"
        print(f"[*] Querying ZIP archive metadata: {zip_url}")

        resp = self.session.get(zip_url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            self._print_auth_guidance()
            resp.raise_for_status()
        else:
            resp.raise_for_status()
        return {}

    def get_download_url(self) -> str:
        """
        Retrieves the direct download URL by querying the redirect endpoint:
        GET https://api.data.mendeley.com/datasets/{id}/zip/file_downloaded?version={version}
        """
        dl_url = f"{API_BASE}/datasets/{self.dataset_id}/zip/file_downloaded?version={self.version}"
        print(f"[*] Resolving download redirect: {dl_url}")

        resp = self.session.get(dl_url, allow_redirects=False, timeout=15)
        if resp.status_code in (301, 302, 303, 307, 308):
            redirect_url = resp.headers.get("Location")
            print(f"[+] Direct download location resolved: {redirect_url[:80]}...")
            return redirect_url
        elif resp.status_code == 200:
            # Direct response
            return dl_url
        elif resp.status_code == 401:
            self._print_auth_guidance()
            resp.raise_for_status()
        else:
            resp.raise_for_status()
        return ""

    def _print_auth_guidance(self):
        """Displays clear instructions when authentication is required."""
        print("\n" + "=" * 70)
        print("          AUTHENTICATION REQUIRED BY MENDELEY DATA API          ")
        print("=" * 70)
        print("Mendeley Data API requires an OAuth2 Bearer token for automated downloads.")
        print("\nTo configure authentication, choose ONE of the following options:")
        print("\nOption A: Set an Access Token (Recommended)")
        print("  1. Obtain your personal access token from https://dev.mendeley.com/myapps.html")
        print("  2. In your PowerShell terminal, set the environment variable:")
        print("     $env:MENDELEY_ACCESS_TOKEN=\"your_access_token_here\"")
        print("\nOption B: Set Client ID and Secret (Client Credentials)")
        print("  1. Create a Mendeley developer application at https://dev.mendeley.com/myapps.html")
        print("  2. In your PowerShell terminal, set:")
        print("     $env:MENDELEY_CLIENT_ID=\"your_client_id\"")
        print("     $env:MENDELEY_CLIENT_SECRET=\"your_client_secret\"")
        print("\nOption C: Pass token directly via CLI flag:")
        print("  python src/download_dataset.py --token \"your_access_token\"")
        print("=" * 70 + "\n")

    def download(
        self,
        direct_url: Optional[str] = None,
        force: bool = False,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> bool:
        """
        Executes streaming chunked download with HTTP Range resume and automatic retries.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve ZIP metadata and download redirect
        print("\n[*] Step 1/3: Fetching archive metadata and download URL...")
        if direct_url:
            download_url = direct_url
            zip_meta = {}
            print(f"[+] Using direct download URL: {download_url[:80]}...")
        else:
            try:
                zip_meta = self.get_zip_metadata()
            except Exception as e:
                print(f"[!] Could not retrieve ZIP metadata: {e}")
                zip_meta = {}

            try:
                download_url = self.get_download_url()
            except Exception as e:
                print(f"[!] Download redirect resolution failed: {e}")
                return False

        expected_size = zip_meta.get("size")
        expected_hash = zip_meta.get("sha256") or zip_meta.get("hash")
        expected_name = zip_meta.get("name") or self.archive_name

        print(f"\n[*] Step 2/3: Checking target destination: {self.destination_path}")
        if self.destination_path.exists():
            current_size = self.destination_path.stat().st_size
            if expected_size and current_size == expected_size and not force:
                print(f"[+] Archive already exists and matches expected size ({current_size:,} bytes).")
                print("    Skipping download. Use --force to re-download or --verify-only to validate.")
                return True
            elif current_size > 0:
                print(f"[*] Partial file detected ({current_size / (1024**2):.2f} MB). Will attempt resume.")
            elif force:
                print("[*] --force specified. Overwriting existing file.")
                self.destination_path.unlink()

        # Step 3: Resilient Chunked Streaming Download
        print(f"\n[*] Step 3/3: Initiating streaming download...")
        success = False
        attempt = 0

        while attempt < max_retries and not success:
            attempt += 1
            existing_size = self.destination_path.stat().st_size if self.destination_path.exists() else 0

            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                print(f"[*] Attempt {attempt}/{max_retries}: Resuming from byte {existing_size:,}...")
            else:
                print(f"[*] Attempt {attempt}/{max_retries}: Starting new download...")

            try:
                with self.session.get(download_url, headers=headers, stream=True, timeout=30) as resp:
                    # Check status
                    if resp.status_code == 416:
                        # Range Not Satisfiable -> File is already complete
                        print("[+] Server returned 416: File already completely downloaded.")
                        success = True
                        break

                    if resp.status_code not in (200, 206):
                        print(f"[!] Server returned unexpected HTTP {resp.status_code}: {resp.reason}")
                        time.sleep(backoff_factor ** attempt)
                        continue

                    is_resume = (resp.status_code == 206)
                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        total_bytes = existing_size + int(content_length) if is_resume else int(content_length)
                    else:
                        total_bytes = expected_size or int(EXPECTED_APPROX_SIZE_GB * 1024**3)

                    write_mode = "ab" if is_resume else "wb"
                    if not is_resume and existing_size > 0:
                        print("[!] Server does not support Range requests. Restarting from byte 0.")
                        existing_size = 0

                    with open(self.destination_path, write_mode) as f, tqdm(
                        total=total_bytes,
                        initial=existing_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=self.archive_name,
                        ncols=80,
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                    final_size = self.destination_path.stat().st_size
                    if expected_size and final_size < expected_size:
                        print(f"\n[!] Incomplete download ({final_size:,} / {expected_size:,} bytes). Retrying...")
                        time.sleep(backoff_factor ** attempt)
                        continue

                    success = True

            except (requests.exceptions.RequestException, IOError) as err:
                print(f"\n[!] Network error on attempt {attempt}: {err}")
                sleep_time = backoff_factor ** attempt
                print(f"[*] Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

        if not success:
            print(f"\n[ERROR] Download failed after {max_retries} attempts.")
            return False

        print("\n[+] Download finished successfully!")
        return self.verify_archive(expected_size=expected_size, expected_hash=expected_hash)

    def verify_archive(
        self,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
    ) -> bool:
        """
        Validates the downloaded archive:
        1. Non-empty file check.
        2. File size matching.
        3. SHA-256 calculation.
        4. ZipFile integrity check (CRC-32 headers).
        """
        print("\n" + "=" * 65)
        print("          KneeWise Archive Verification Engine                 ")
        print("=" * 65)
        print(f"[*] Archive Path: {self.destination_path}")

        if not self.destination_path.exists():
            print(f"[!] File not found: {self.destination_path}")
            return False

        actual_size = self.destination_path.stat().st_size
        print(f"[*] File Size:    {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)")

        if actual_size == 0:
            print("[!] Verification failed: Zero-byte archive.")
            return False

        if expected_size:
            if actual_size == expected_size:
                print(f"[+] File Size:    MATCHES expected size ({expected_size:,} bytes)")
            else:
                print(f"[!] Size Mismatch: Expected {expected_size:,} bytes, found {actual_size:,} bytes.")
                return False

        # Compute SHA-256
        print("[*] Computing SHA-256 checksum (this may take a minute for ~6.68 GB)...")
        hasher = hashlib.sha256()
        with open(self.destination_path, "rb") as f, tqdm(
            total=actual_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Hashing SHA-256",
            ncols=80,
        ) as pbar:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
                pbar.update(len(chunk))

        calc_hash = hasher.hexdigest()
        print(f"[*] Calculated SHA-256: {calc_hash}")

        if expected_hash:
            if calc_hash.lower() == expected_hash.lower():
                print(f"[+] SHA-256 Hash: MATCHES expected hash!")
            else:
                print(f"[!] Hash Mismatch: Expected {expected_hash}, calculated {calc_hash}")
                return False
        else:
            print("[*] No official hash provided by API endpoint for automated comparison.")

        # Non-destructive zip integrity test
        print("[*] Testing ZIP archive structural integrity (testzip)...")
        try:
            with zipfile.ZipFile(self.destination_path, "r") as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    print(f"[!] ZIP corrupted! First damaged file: {bad_file}")
                    return False
                file_list = zf.namelist()
                print(f"[+] ZIP Integrity: VALID (Contains {len(file_list):,} internal files)")
                print("    Sample files inside archive:")
                for fname in file_list[:5]:
                    print(f"      - {fname}")
        except zipfile.BadZipFile as e:
            print(f"[!] Bad ZIP file error: {e}")
            return False

        print("\n" + "=" * 65)
        print("[PASSED] ARCHIVE FULLY VERIFIED AND READY FOR EXTRACTION!")
        print("=" * 65)
        return True


def display_metadata_report(meta: Dict[str, Any]):
    """Pretty-prints dataset metadata in structured readable format."""
    print("\n" + "=" * 65)
    print("      Mendeley Data API — Dataset Metadata Report             ")
    print("=" * 65)
    print(f"Title:         {meta.get('name')}")
    print(f"Dataset ID:    {meta.get('id')}")
    print(f"Version:       {meta.get('version')}")
    print(f"DOI:           {meta.get('doi', {}).get('id') if isinstance(meta.get('doi'), dict) else meta.get('doi')}")
    print(f"Published On:  {meta.get('publish_date')}")

    contributors = meta.get("contributors", [])
    if contributors:
        author_names = [f"{c.get('first_name', '')} {c.get('last_name', '')}".strip() for c in contributors]
        print(f"Author(s):     {', '.join(author_names)}")

    license_info = meta.get("data_licence") or meta.get("licence", {})
    if isinstance(license_info, dict):
        print(f"License:       {license_info.get('short_name', 'CC BY 4.0')} ({license_info.get('url', '')})")

    desc = meta.get("description") or meta.get("method") or ""
    print(f"\nDescription / Provenance:")
    print(f"  {desc.strip()}")

    categories = meta.get("categories", [])
    if categories:
        cat_labels = [c.get("label") for c in categories if isinstance(c, dict) and "label" in c]
        print(f"\nCategories:    {', '.join(cat_labels)}")

    files = meta.get("files", [])
    print(f"\nFiles count:   {len(files)}")
    for f in files[:5]:
        print(f"  - {f.get('name')} ({f.get('size', 0):,} bytes)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="KneeWise Mendeley Data API Downloader")
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID, help="Mendeley dataset ID")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Dataset version")
    parser.add_argument("--output_dir", type=str, default=str(RAW_DATA_DIR), help="Output directory")
    parser.add_argument("--filename", type=str, default=DEFAULT_ARCHIVE_NAME, help="Archive filename")
    parser.add_argument("--url", type=str, default=None, help="Direct download URL (bypasses API redirect)")
    parser.add_argument("--token", type=str, default=None, help="Mendeley OAuth2 Access Token")
    parser.add_argument("--client_id", type=str, default=None, help="Mendeley OAuth2 Client ID")
    parser.add_argument("--client_secret", type=str, default=None, help="Mendeley OAuth2 Client Secret")
    parser.add_argument("--metadata-only", action="store_true", help="Inspect metadata without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing archive without downloading")
    parser.add_argument("--force", action="store_true", help="Overwrite existing completed archive")
    args = parser.parse_args()

    downloader = MendeleyDownloader(
        dataset_id=args.dataset_id,
        version=args.version,
        output_dir=Path(args.output_dir),
        archive_name=args.filename,
        access_token=args.token,
        client_id=args.client_id,
        client_secret=args.client_secret,
    )

    downloader.authenticate()

    if args.verify_only:
        success = downloader.verify_archive()
        sys.exit(0 if success else 1)

    if args.metadata_only:
        meta = downloader.get_dataset_metadata()
        display_metadata_report(meta)
        sys.exit(0)

    # Full download workflow
    meta = downloader.get_dataset_metadata()
    display_metadata_report(meta)

    success = downloader.download(direct_url=args.url, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
