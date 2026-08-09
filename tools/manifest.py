"""
Manifest Manager for INCLUDE-50 Remote ZIP Pipeline.
Maintains persistent state in data/metadata/download_manifest.csv.
Supports retry, interruption, resume, and filesystem reconciliation.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from tools.validation import calculate_crc32, is_safe_path

MANIFEST_COLUMNS = [
    "video_path",
    "label",
    "split",
    "archive",
    "archive_url",
    "status",
    "compressed_size",
    "original_size",
    "crc32",
    "local_header_offset",
    "data_offset",
    "output_path",
    "attempts",
    "error",
]

# Status constants
STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_COMPLETED = "completed"
STATUS_VERIFIED = "verified"
STATUS_FAILED = "failed"


class ManifestManager:
    def __init__(
        self,
        manifest_path: Path = Path("data/metadata/download_manifest.csv"),
        selected_csv: Path = Path("data/metadata/selected_videos.csv"),
        video_base_dir: Path = Path("data/videos"),
    ):
        self.manifest_path = Path(manifest_path)
        self.selected_csv = Path(selected_csv)
        self.video_base_dir = Path(video_base_dir)
        self._lock = threading.Lock()

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load_or_initialize()
        self._reconcile_filesystem()

    def _load_or_initialize(self) -> pd.DataFrame:
        """
        Loads existing download_manifest.csv or creates a new manifest from selected_videos.csv.
        """
        if self.manifest_path.exists():
            df = pd.read_csv(self.manifest_path, dtype=str)  # Read everything as str first
            # Ensure all columns exist
            for col in MANIFEST_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            # Coerce numeric columns safely
            for num_col in ("compressed_size", "original_size", "crc32", "local_header_offset", "data_offset", "attempts"):
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0).astype(object)
            df["error"] = df["error"].fillna("")
            return df[MANIFEST_COLUMNS]

        # Initialize from selected_videos.csv
        if not self.selected_csv.exists():
            raise FileNotFoundError(f"Authoritative selection file not found: {self.selected_csv}")

        selected_df = pd.read_csv(self.selected_csv)
        rows = []

        for _, row in selected_df.iterrows():
            vpath = str(row["video_path"]).replace("\\", "/")
            label = str(row["label"])
            split = str(row["split"])
            archive = str(row["archive"])
            archive_url = str(row["archive_url"])
            filename = Path(vpath).name

            output_path = (self.video_base_dir / split / label / filename).as_posix()

            rows.append(
                {
                    "video_path": vpath,
                    "label": label,
                    "split": split,
                    "archive": archive,
                    "archive_url": archive_url,
                    "status": STATUS_PENDING,
                    "compressed_size": 0,
                    "original_size": 0,
                    "crc32": 0,
                    "local_header_offset": 0,
                    "data_offset": 0,
                    "output_path": output_path,
                    "attempts": 0,
                    "error": "",
                }
            )

        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        df.to_csv(self.manifest_path, index=False)
        return df

    def _reconcile_filesystem(self):
        """
        Reconciles manifest status with local files on disk.
        If a file exists locally and CRC matches, marks it verified.
        If a file is corrupt/0-bytes, marks it pending and removes it.
        """
        updated = False
        with self._lock:
            for idx, row in self.df.iterrows():
                out_path = Path(row["output_path"])
                status = row["status"]
                expected_crc = int(row["crc32"]) if row["crc32"] else 0

                if out_path.exists() and out_path.stat().st_size > 0:
                    if expected_crc > 0:
                        actual_crc = calculate_crc32(out_path)
                        if actual_crc == expected_crc:
                            if status != STATUS_VERIFIED:
                                self.df.at[idx, "status"] = STATUS_VERIFIED
                                updated = True
                        else:
                            # Corrupted output file; delete and mark pending
                            try:
                                out_path.unlink()
                            except OSError:
                                pass
                            self.df.at[idx, "status"] = STATUS_PENDING
                            self.df.at[idx, "error"] = "CRC mismatch on existing file (reconciled)"
                            updated = True
                    else:
                        # File exists but CRC wasn't recorded yet; keep existing status or verify if completed
                        if status == STATUS_COMPLETED:
                            self.df.at[idx, "status"] = STATUS_VERIFIED
                            updated = True

            if updated:
                self.df.to_csv(self.manifest_path, index=False)

    def save(self):
        """
        Saves current manifest state to disk.
        """
        with self._lock:
            self.df.to_csv(self.manifest_path, index=False)

    def update_video(self, video_path: str, **kwargs):
        """
        Updates fields for a specific video_path and persists manifest immediately.
        """
        vpath_norm = str(video_path).replace("\\", "/")
        with self._lock:
            matches = self.df[self.df["video_path"] == vpath_norm]
            if len(matches) == 0:
                return

            idx = matches.index[0]
            for key, val in kwargs.items():
                if key in MANIFEST_COLUMNS:
                    # Always store as Python native type to avoid pandas dtype coercion issues
                    if val is None:
                        val = ""
                    self.df.at[idx, key] = val

            self.df.to_csv(self.manifest_path, index=False)

    def get_pending_videos(self, retry_failed: bool = False) -> List[Dict[str, Any]]:
        """
        Returns list of video dicts that need processing.
        """
        with self._lock:
            target_statuses = [STATUS_PENDING, STATUS_DOWNLOADING]
            if retry_failed:
                target_statuses.append(STATUS_FAILED)

            pending_df = self.df[self.df["status"].isin(target_statuses)]
            return pending_df.to_dict("records")

    def get_summary(self) -> Dict[str, int]:
        """
        Returns count dictionary by status.
        """
        with self._lock:
            counts = self.df["status"].value_counts().to_dict()
            return {
                "total": len(self.df),
                "verified": counts.get(STATUS_VERIFIED, 0),
                "completed": counts.get(STATUS_COMPLETED, 0),
                "pending": counts.get(STATUS_PENDING, 0),
                "downloading": counts.get(STATUS_DOWNLOADING, 0),
                "failed": counts.get(STATUS_FAILED, 0),
            }
