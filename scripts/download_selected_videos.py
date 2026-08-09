"""
Main Video Downloader Script for INCLUDE-50 Remote ZIP Pipeline.
Supports --dry-run, --test-one, --retry-failed, --workers options.
Handles remote byte-range ZIP extraction, stream decompression, CRC32 verification,
continuous manifest updates, retries, resume, and graceful interrupt (Ctrl+C).
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple
import requests

from tools.downloader import download_and_extract_video
from tools.manifest import ManifestManager, STATUS_FAILED, STATUS_VERIFIED
from tools.remote_zip import get_central_directory, parse_central_directory
from tools.validation import calculate_crc32, validate_extracted_file

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("downloader")


def run_dry_run(manifest_mgr: ManifestManager):
    """
    Dry run execution: validates selection, inspects remote central directories,
    matches selected videos, and reports statistics without downloading video payloads.
    """
    logger.info("Starting DRY RUN mode...")
    pending = manifest_mgr.get_pending_videos(retry_failed=True)
    summary = manifest_mgr.get_summary()

    logger.info(f"Total selection: {summary['total']} videos")
    logger.info(f"Already verified: {summary['verified']} videos")
    logger.info(f"Pending/Failed videos to process: {len(pending)}")

    archives = {}
    for item in pending:
        arch = item["archive"]
        url = item["archive_url"]
        archives.setdefault(arch, {"url": url, "count": 0})
        archives[arch]["count"] += 1

    logger.info(f"Unique remote ZIP archives involved: {len(archives)}")

    session = requests.Session()
    total_compressed = 0
    total_original = 0
    found_count = 0

    for arch, info in archives.items():
        logger.info(f"Inspecting central directory for archive: {arch} ({info['count']} videos needed)")
        try:
            cd_bytes, cd_offset, entries_count = get_central_directory(info["url"], session=session)
            entries = parse_central_directory(cd_bytes)

            arch_comp = 0
            arch_orig = 0
            arch_found = 0

            group = [p for p in pending if p["archive"] == arch]
            for item in group:
                vpath = item["video_path"]
                entry = entries.get(vpath)
                if entry is None:
                    matches = [e for name, e in entries.items() if name.endswith(Path(vpath).name)]
                    if matches:
                        entry = matches[0]

                if entry:
                    arch_found += 1
                    arch_comp += entry["compressed_size"]
                    arch_orig += entry["original_size"]

            found_count += arch_found
            total_compressed += arch_comp
            total_original += arch_orig
            logger.info(
                f"Archive {arch}: matched {arch_found}/{info['count']} videos. "
                f"Compressed: {arch_comp / 1024**2:.2f} MB, Original: {arch_orig / 1024**2:.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to inspect archive {arch}: {e}")

    logger.info("=" * 60)
    logger.info("DRY RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos matched: {found_count}/{len(pending)}")
    logger.info(f"Total compressed download required: {total_compressed / 1024**3:.2f} GB")
    logger.info(f"Total extracted size: {total_original / 1024**3:.2f} GB")
    logger.info("DRY RUN COMPLETE - No video payloads downloaded.")


def run_test_one(manifest_mgr: ManifestManager, target_vpath: str = "Animals/1. Dog/MVI_3060.MOV"):
    """
    Single-video test mode: extracts and validates one selected video remotely via HTTP Range requests.
    Outputs PASS/FAIL result and does NOT continue with full download.
    """
    logger.info("=" * 60)
    logger.info(f"RUNNING ONE-VIDEO TEST (Target: {target_vpath})")
    logger.info("=" * 60)

    # Find target in manifest
    all_records = manifest_mgr.df.to_dict("records")
    target_item = None

    for item in all_records:
        if target_vpath.lower() in item["video_path"].lower():
            target_item = item
            break

    if target_item is None:
        target_item = all_records[0]
        logger.info(f"Target {target_vpath} not found in selection, using default: {target_item['video_path']}")

    logger.info(f"Testing video: {target_item['video_path']}")
    logger.info(f"Archive:      {target_item['archive']}")
    logger.info(f"Archive URL:  {target_item['archive_url']}")

    session = requests.Session()
    output_path = Path(target_item["output_path"])
    temp_test_output = output_path.parent / f"test_{output_path.name}"

    record_to_test = dict(target_item)
    record_to_test["output_path"] = temp_test_output.as_posix()

    start_time = time.time()
    try:
        res = download_and_extract_video(record_to_test, session=session)
        elapsed = time.time() - start_time

        comp_mb = res["compressed_size"] / (1024**2)
        orig_mb = res["original_size"] / (1024**2)
        crc32_val = res["crc32"]
        out_size = temp_test_output.stat().st_size

        is_valid, val_msg = validate_extracted_file(temp_test_output, expected_crc32=crc32_val)

        logger.info("-" * 60)
        logger.info(f"Archive           : {target_item['archive']}")
        logger.info(f"Video             : {target_item['video_path']}")
        logger.info(f"Compressed bytes  : {res['compressed_size']} ({comp_mb:.2f} MB)")
        logger.info(f"Original size     : {res['original_size']} ({orig_mb:.2f} MB)")
        logger.info(f"CRC32             : 0x{crc32_val:08X}")
        logger.info(f"Output size       : {out_size} bytes")
        logger.info(f"Download time     : {elapsed:.2f} s")
        logger.info(f"Validation        : {val_msg}")

        if is_valid:
            logger.info("\nRemote extraction test: PASS")
        else:
            logger.error(f"\nRemote extraction test: FAIL ({val_msg})")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nRemote extraction test: FAIL with exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up test output file
        if temp_test_output.exists():
            try:
                temp_test_output.unlink()
            except OSError:
                pass


def process_single_video(
    item: Dict[str, Any],
    manifest_mgr: ManifestManager,
    session: requests.Session,
    central_directory_cache: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Downloads and extracts a single video, updating manifest continuously.
    """
    vpath = item["video_path"]
    attempts = item.get("attempts", 0) + 1

    manifest_mgr.update_video(vpath, status="downloading", attempts=attempts)

    try:
        res = download_and_extract_video(
            item,
            session=session,
            central_directory_cache=central_directory_cache,
        )

        manifest_mgr.update_video(
            vpath,
            status=STATUS_VERIFIED,
            compressed_size=res["compressed_size"],
            original_size=res["original_size"],
            crc32=res["crc32"],
            local_header_offset=res["local_header_offset"],
            data_offset=res["data_offset"],
            error="",
        )
        return True, vpath
    except Exception as e:
        logger.error(f"Failed video '{vpath}': {e}")
        manifest_mgr.update_video(vpath, status=STATUS_FAILED, error=str(e))
        return False, vpath


def main():
    parser = argparse.ArgumentParser(description="INCLUDE-50 Remote ZIP Video Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Inspect archives and metadata without downloading videos")
    parser.add_argument("--test-one", action="store_true", help="Extract and validate one video to test pipeline then exit")
    parser.add_argument("--retry-failed", action="store_true", help="Retry videos marked as failed in manifest")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent download workers (default: 1)")

    args = parser.parse_args()

    manifest_mgr = ManifestManager()

    if args.dry_run:
        run_dry_run(manifest_mgr)
        return

    if args.test_one:
        run_test_one(manifest_mgr)
        return

    pending_videos = manifest_mgr.get_pending_videos(retry_failed=args.retry_failed)
    summary = manifest_mgr.get_summary()

    logger.info("=" * 60)
    logger.info("INCLUDE-50 REMOTE ZIP VIDEO DOWNLOADER")
    logger.info("=" * 60)
    logger.info(f"Total selection   : {summary['total']}")
    logger.info(f"Already verified  : {summary['verified']}")
    logger.info(f"Pending to process: {len(pending_videos)}")
    logger.info(f"Workers           : {args.workers}")

    if not pending_videos:
        logger.info("All selected videos are already downloaded and verified!")
        return

    cd_cache: Dict[str, Dict[str, Any]] = {}
    completed_count = summary["verified"]
    failed_count = summary["failed"]
    total_videos = summary["total"]

    start_time = time.time()

    try:
        if args.workers == 1:
            session = requests.Session()
            for idx, item in enumerate(pending_videos, 1):
                cur_num = completed_count + 1
                logger.info(
                    f"[{cur_num}/{total_videos}] Label: {item['label']} | Video: {item['video_path']} | Archive: {item['archive']}"
                )
                success, _ = process_single_video(item, manifest_mgr, session, cd_cache)
                if success:
                    completed_count += 1
                else:
                    failed_count += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                sessions = [requests.Session() for _ in range(args.workers)]
                futures = {}
                for idx, item in enumerate(pending_videos):
                    sess = sessions[idx % args.workers]
                    fut = executor.submit(process_single_video, item, manifest_mgr, sess, cd_cache)
                    futures[fut] = item

                for fut in as_completed(futures):
                    success, vpath = fut.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1

    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPTED] KeyboardInterrupt received. Saving manifest state...")
        manifest_mgr.save()
        logger.info("Manifest saved successfully. Re-run 'python scripts/download_selected_videos.py' to resume.")
        sys.exit(130)

    elapsed = time.time() - start_time
    final_summary = manifest_mgr.get_summary()

    logger.info("=" * 60)
    logger.info("DOWNLOAD PROCESS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total elapsed time : {elapsed / 60:.2f} minutes")
    logger.info(f"Verified videos    : {final_summary['verified']}")
    logger.info(f"Failed videos      : {final_summary['failed']}")
    logger.info(f"Pending videos     : {final_summary['pending']}")


if __name__ == "__main__":
    main()
