"""
Remote ZIP and HTTP Range Request Utilities for INCLUDE-50 Pipeline.
Handles binary ZIP parsing, ZIP64 extensions, central directory parsing,
local header reading, and byte-range requests.
"""

import io
import struct
import time
from typing import Any, Dict, Optional, Tuple
import requests

# Default settings
DEFAULT_TIMEOUT = (15, 120)  # (connect, read)
MAX_RETRIES = 5
BACKOFF_FACTOR = 3  # Backoff: 3s, 6s, 12s, 24s, 48s


def range_request(
    url: str,
    start: int,
    end: int,
    session: Optional[requests.Session] = None,
    max_retries: int = MAX_RETRIES,
    timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
) -> requests.Response:
    """
    Perform an HTTP byte-range request with exponential backoff retries.
    Validates HTTP 206 status. Fails safely if HTTP 200 is returned.
    """
    sess = session or requests.Session()
    headers = {"Range": f"bytes={start}-{end}"}

    for attempt in range(1, max_retries + 1):
        try:
            response = sess.get(url, headers=headers, timeout=timeout)

            # Strict Range validation: Zenodo MUST return 206 Partial Content
            if response.status_code == 200:
                raise RuntimeError(
                    f"Server returned HTTP 200 (Full Content) instead of HTTP 206 Range Response for {url}"
                )

            response.raise_for_status()

            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206 Partial Content, got HTTP {response.status_code}"
                )

            return response

        except (requests.RequestException, RuntimeError) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"HTTP Range request failed after {max_retries} attempts: {e}"
                ) from e

            # Exponential backoff: 3, 6, 12, 24, 48 seconds
            wait_time = BACKOFF_FACTOR * (2 ** (attempt - 1))
            time.sleep(wait_time)


def get_zip_tail(
    url: str,
    tail_size: int = 65536,
    session: Optional[requests.Session] = None,
    max_retries: int = MAX_RETRIES,
    timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
) -> bytes:
    """
    Fetch the last tail_size bytes of a remote ZIP file.
    """
    sess = session or requests.Session()
    headers = {"Range": f"bytes=-{tail_size}"}

    for attempt in range(1, max_retries + 1):
        try:
            response = sess.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                raise RuntimeError(
                    f"Server returned HTTP 200 instead of 206 when requesting tail of {url}"
                )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206 for tail request, got {response.status_code}"
                )
            return response.content
        except (requests.RequestException, RuntimeError) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch ZIP tail after {max_retries} attempts: {e}"
                ) from e
            wait_time = BACKOFF_FACTOR * (2 ** (attempt - 1))
            time.sleep(wait_time)


def parse_eocd(tail_data: bytes) -> Dict[str, Any]:
    """
    Locate and parse the End Of Central Directory (EOCD) record from tail bytes.
    Signature: PK\\x05\\x06
    """
    signature = b"PK\x05\x06"
    position = tail_data.rfind(signature)

    if position == -1:
        raise ValueError("End Of Central Directory (EOCD) record (PK\\x05\\x06) not found in tail buffer.")

    # EOCD Layout (22 bytes fixed):
    # 4s (sig), 2H (disk, cd_start_disk), 2H (entries_disk, total_entries), 2L (cd_size, cd_offset), H (comment_len)
    fields = struct.unpack_from("<4s4H2LH", tail_data, position)
    (
        sig,
        disk,
        cd_start_disk,
        entries_disk,
        total_entries,
        cd_size,
        cd_offset,
        comment_len,
    ) = fields

    is_zip64 = (
        total_entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    )

    return {
        "eocd_pos": position,
        "total_entries": total_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "comment_len": comment_len,
        "is_zip64": is_zip64,
    }


def parse_zip64_locator_and_eocd(
    tail_data: bytes,
    eocd_pos: int,
    url: str,
    session: Optional[requests.Session] = None,
) -> Tuple[int, int, int]:
    """
    Locate and parse ZIP64 locator and ZIP64 EOCD record.
    Returns (total_entries, cd_size, cd_offset).
    """
    # ZIP64 Locator signature: PK\x06\x07 (20 bytes long, immediately precedes EOCD)
    locator_sig = b"PK\x06\x07"
    locator_pos = tail_data.rfind(locator_sig, 0, eocd_pos)

    if locator_pos == -1:
        raise ValueError("ZIP64 locator (PK\\x06\\x07) not found in tail buffer.")

    # Locator layout: 4s (sig), I (disk_zip64_eocd), Q (zip64_eocd_offset), I (total_disks)
    _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sIQI", tail_data, locator_pos)

    # ZIP64 EOCD Record signature: PK\x06\x06 (56 bytes fixed)
    # Check if ZIP64 EOCD record is already in our tail_data
    # Determine absolute remote file size if possible or fetch directly
    zip64_eocd_bytes = range_request(
        url,
        zip64_eocd_offset,
        zip64_eocd_offset + 55,
        session=session,
    ).content

    sig = zip64_eocd_bytes[:4]
    if sig != b"PK\x06\x06":
        raise ValueError(f"Invalid ZIP64 EOCD signature: {sig}")

    # ZIP64 EOCD layout (56 bytes):
    # 4s (sig), Q (record_size), 2H (version_made, version_needed), 2I (disk, cd_start_disk),
    # 2Q (entries_disk, total_entries), 2Q (cd_size, cd_offset)
    fields = struct.unpack_from("<4sQ2H2I4Q", zip64_eocd_bytes, 0)
    total_entries = fields[7]
    cd_size = fields[8]
    cd_offset = fields[9]

    return total_entries, cd_size, cd_offset


def get_central_directory(
    url: str, session: Optional[requests.Session] = None
) -> Tuple[bytes, int, int]:
    """
    Fetches the ZIP central directory using HTTP Range requests.
    Returns (central_directory_bytes, cd_offset, total_entries).
    """
    sess = session or requests.Session()
    # Try fetching 64KB tail first; expand to 128KB if needed
    tail_size = 65536
    tail_data = get_zip_tail(url, tail_size=tail_size, session=sess)

    try:
        eocd_info = parse_eocd(tail_data)
    except ValueError:
        # If tail was too small for comment/EOCD, retry with 128KB tail
        tail_data = get_zip_tail(url, tail_size=131072, session=sess)
        eocd_info = parse_eocd(tail_data)

    if eocd_info["is_zip64"]:
        total_entries, cd_size, cd_offset = parse_zip64_locator_and_eocd(
            tail_data, eocd_info["eocd_pos"], url, session=sess
        )
    else:
        total_entries = eocd_info["total_entries"]
        cd_size = eocd_info["cd_size"]
        cd_offset = eocd_info["cd_offset"]

    cd_end = cd_offset + cd_size - 1
    cd_response = range_request(url, cd_offset, cd_end, session=sess)
    return cd_response.content, cd_offset, total_entries


def parse_central_directory(cd_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """
    Parses ZIP Central Directory entries (PK\\x01\\x02).
    Returns dictionary mapping normalized file paths ('\\' -> '/') to entry details.
    """
    entries = {}
    pos = 0
    cd_len = len(cd_bytes)

    while pos + 46 <= cd_len:
        sig = cd_bytes[pos : pos + 4]
        if sig != b"PK\x01\x02":
            break

        # Fixed Header (46 bytes):
        # 0: sig (4B)
        # 4: version_made (2B), 6: version_needed (2B), 8: flag_bits (2B), 10: compression_method (2B)
        # 12: last_mod_time (2B), 14: last_mod_date (2B), 16: crc32 (4B)
        # 20: compressed_size (4B), 24: uncompressed_size (4B)
        # 28: filename_length (2B), 30: extra_length (2B), 32: comment_length (2B)
        # 34: disk_number_start (2B), 36: internal_attr (2B), 38: external_attr (4B), 42: local_header_offset (4B)

        compression_method = struct.unpack_from("<H", cd_bytes, pos + 10)[0]
        crc32 = struct.unpack_from("<I", cd_bytes, pos + 16)[0]
        compressed_size = struct.unpack_from("<I", cd_bytes, pos + 20)[0]
        uncompressed_size = struct.unpack_from("<I", cd_bytes, pos + 24)[0]
        filename_length = struct.unpack_from("<H", cd_bytes, pos + 28)[0]
        extra_length = struct.unpack_from("<H", cd_bytes, pos + 30)[0]
        comment_length = struct.unpack_from("<H", cd_bytes, pos + 32)[0]
        local_header_offset = struct.unpack_from("<I", cd_bytes, pos + 42)[0]

        fn_start = pos + 46
        fn_end = fn_start + filename_length
        filename_raw = cd_bytes[fn_start:fn_end]
        filename = filename_raw.decode("utf-8", errors="replace")

        # ZIP internal paths use forward slashes
        normalized_filename = filename.replace("\\", "/")

        extra_start = fn_end
        extra_end = extra_start + extra_length
        extra_bytes = cd_bytes[extra_start:extra_end]

        # Parse ZIP64 Extra Field (Header ID 0x0001) if needed
        if (
            uncompressed_size == 0xFFFFFFFF
            or compressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            extra_pos = 0
            while extra_pos + 4 <= len(extra_bytes):
                header_id, data_size = struct.unpack_from("<HH", extra_bytes, extra_pos)
                extra_pos += 4
                if header_id == 0x0001:  # ZIP64 Extended Information Extra Field
                    field_pos = extra_pos
                    if uncompressed_size == 0xFFFFFFFF and field_pos + 8 <= extra_pos + data_size:
                        uncompressed_size = struct.unpack_from("<Q", extra_bytes, field_pos)[0]
                        field_pos += 8
                    if compressed_size == 0xFFFFFFFF and field_pos + 8 <= extra_pos + data_size:
                        compressed_size = struct.unpack_from("<Q", extra_bytes, field_pos)[0]
                        field_pos += 8
                    if local_header_offset == 0xFFFFFFFF and field_pos + 8 <= extra_pos + data_size:
                        local_header_offset = struct.unpack_from("<Q", extra_bytes, field_pos)[0]
                        field_pos += 8
                    break
                extra_pos += data_size

        entries[normalized_filename] = {
            "filename": normalized_filename,
            "compression_method": compression_method,
            "crc32": crc32,
            "compressed_size": compressed_size,
            "original_size": uncompressed_size,
            "local_header_offset": local_header_offset,
        }

        pos = fn_end + extra_length + comment_length

    return entries


def get_local_file_data_offset(
    url: str, local_header_offset: int, session: Optional[requests.Session] = None
) -> int:
    """
    Fetches the 30-byte Local File Header at local_header_offset via range request,
    parses local filename length and extra length, and calculates exact compressed data start offset.
    Signature: PK\\x03\\x04
    """
    header_bytes = range_request(
        url,
        local_header_offset,
        local_header_offset + 29,
        session=session,
    ).content

    sig = header_bytes[:4]
    if sig != b"PK\x03\x04":
        raise ValueError(
            f"Invalid Local File Header signature (expected PK\\x03\\x04, got {sig}) at offset {local_header_offset}"
        )

    # Local File Header layout (30 bytes):
    # 0: sig (4B), 4: version (2B), 6: flag (2B), 8: compression (2B), 10: mod_time (2B), 12: mod_date (2B),
    # 14: crc32 (4B), 18: compressed_size (4B), 22: uncompressed_size (4B),
    # 26: filename_len (2B), 28: extra_len (2B)
    filename_len = struct.unpack_from("<H", header_bytes, 26)[0]
    extra_len = struct.unpack_from("<H", header_bytes, 28)[0]

    compressed_data_start = local_header_offset + 30 + filename_len + extra_len
    return compressed_data_start
