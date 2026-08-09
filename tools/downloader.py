"""
Downloader and Decompressor Module for INCLUDE-50 Remote ZIP Pipeline.
Handles exact byte-range downloads of compressed video data, local Deflate/Store decompression,
CRC32 validation, memory-safe chunked streaming, temporary file handling, and safe output writing.
"""

import logging
from pathlib import Path

from typing import Any, Dict, Optional
import zlib
import requests

from tools.remote_zip import get_central_directory, get_local_file_data_offset, parse_central_directory, range_request
from tools.validation import calculate_crc32, is_safe_path

logger = logging.getLogger(__name__)


def download_and_extract_video(
    video_record: Dict[str, Any],
    session: Optional[requests.Session] = None,
    base_video_dir: Path = Path("data/videos"),
    central_directory_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Downloads and extracts a single video file from a remote ZIP archive via HTTP Range requests.

    Steps:
    1. Inspect remote central directory (cached if provided).
    2. Determine local header offset and compressed data offset.
    3. Stream compressed byte range into a .comp temporary file.
    4. Decompress into a .part temporary file (supporting Deflate & Store).
    5. Calculate CRC32 checksum during decompression.
    6. Validate CRC32 checksum and non-zero size.
    7. Atomically rename .part file to final .MOV path inside data/videos/{split}/{label}/.
    """
    sess = session or requests.Session()

    vpath = str(video_record["video_path"]).replace("\\", "/")
    archive_url = str(video_record["archive_url"])
    output_path = Path(video_record["output_path"])

    # Path traversal protection
    if not is_safe_path(base_video_dir, output_path):
        raise ValueError(f"Security error: Output path {output_path} escapes base directory {base_video_dir}")

    # 1. Access Central Directory (use cache if available)
    if central_directory_cache is not None and archive_url in central_directory_cache:
        cd_entries = central_directory_cache[archive_url]
    else:
        cd_bytes, _, _ = get_central_directory(archive_url, session=sess)
        cd_entries = parse_central_directory(cd_bytes)
        if central_directory_cache is not None:
            central_directory_cache[archive_url] = cd_entries

    # Match entry in Central Directory
    entry = cd_entries.get(vpath)
    if entry is None:
        # Try matching without leading directories if full path mismatch occurs
        matches = [e for name, e in cd_entries.items() if name.endswith(Path(vpath).name)]
        if matches:
            entry = matches[0]
        else:
            raise ValueError(f"Video '{vpath}' not found in remote central directory of archive '{video_record['archive']}'")

    local_header_offset = entry["local_header_offset"]
    compressed_size = entry["compressed_size"]
    original_size = entry["original_size"]
    expected_crc32 = entry["crc32"]
    compression_method = entry["compression_method"]

    # 2. Determine exact compressed data start offset
    data_offset = get_local_file_data_offset(archive_url, local_header_offset, session=sess)
    compressed_start = data_offset
    compressed_end = data_offset + compressed_size - 1

    # Temporary paths
    temp_comp_path = output_path.with_suffix(".MOV.comp")
    temp_part_path = output_path.with_suffix(".MOV.part")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Stream compressed byte range into temporary file
        response = range_request(archive_url, compressed_start, compressed_end, session=sess)

        with open(temp_comp_path, "wb") as f_comp:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f_comp.write(chunk)

        # 4. Decompress and calculate CRC32 on the fly
        crc = 0
        with open(temp_comp_path, "rb") as f_comp, open(temp_part_path, "wb") as f_part:
            if compression_method == 0:  # Store (no compression)
                while True:
                    chunk = f_comp.read(65536)
                    if not chunk:
                        break
                    crc = zlib.crc32(chunk, crc)
                    f_part.write(chunk)

            elif compression_method == 8:  # Deflate
                decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                while True:
                    chunk = f_comp.read(65536)
                    if not chunk:
                        break
                    decomp_chunk = decompressor.decompress(chunk)
                    if decomp_chunk:
                        crc = zlib.crc32(decomp_chunk, crc)
                        f_part.write(decomp_chunk)
                # Flush remaining bytes
                flush_chunk = decompressor.flush()
                if flush_chunk:
                    crc = zlib.crc32(flush_chunk, crc)
                    f_part.write(flush_chunk)

            else:
                raise ValueError(f"Unsupported compression method: {compression_method} for video {vpath}")

        computed_crc = crc & 0xFFFFFFFF

        # 5. Validate CRC32 and file size
        if computed_crc != expected_crc32:
            raise RuntimeError(
                f"CRC32 mismatch for '{vpath}': expected 0x{expected_crc32:08X}, got 0x{computed_crc:08X}"
            )

        extracted_size = temp_part_path.stat().st_size
        if extracted_size == 0:
            raise RuntimeError(f"Extracted video file is 0 bytes for '{vpath}'")

        if original_size > 0 and extracted_size != original_size:
            raise RuntimeError(
                f"Decompressed size mismatch for '{vpath}': expected {original_size} bytes, got {extracted_size} bytes"
            )

        # 6. Atomic move to final location
        if output_path.exists():
            output_path.unlink()
        temp_part_path.rename(output_path)

        return {
            "compressed_size": compressed_size,
            "original_size": extracted_size,
            "crc32": expected_crc32,
            "local_header_offset": local_header_offset,
            "data_offset": data_offset,
            "output_path": output_path.as_posix(),
        }

    finally:
        # Cleanup temporary files
        if temp_comp_path.exists():
            try:
                temp_comp_path.unlink()
            except OSError:
                pass
        if temp_part_path.exists():
            try:
                temp_part_path.unlink()
            except OSError:
                pass
