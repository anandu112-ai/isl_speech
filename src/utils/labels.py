"""
Utility functions for text normalization, dataset label parsing, and speech text formatting.
Includes deterministic 36-class mapping for ISL Alphabets (A-Z) and Digits (0-9).
"""

import json
import re
from pathlib import Path
from typing import Dict, List


ALPHABET_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z (26)
DIGIT_CLASSES = [str(i) for i in range(10)]                          # 0-9 (10)
TARGET_36_CLASSES = ALPHABET_CLASSES + DIGIT_CLASSES                 # Total 36


def get_alphabet_digit_classes() -> List[str]:
    """Returns the ordered list of 36 target classes: A-Z (0-25) and 0-9 (26-35)."""
    return list(TARGET_36_CLASSES)


def create_deterministic_alphabet_label_map() -> Dict[str, int]:
    """
    Creates deterministic label mapping dictionary:
    A=0, B=1, ..., Z=25, 0=26, 1=27, ..., 9=35
    """
    return {label: idx for idx, label in enumerate(TARGET_36_CLASSES)}


def normalize_alphabet_label(label: str) -> str:
    """
    Normalizes arbitrary string input to standard target class ('A'-'Z' or '0'-'9').
    Examples:
        'a' -> 'A', 'A' -> 'A', '0' -> '0', 'zero' -> '0', '1. A' -> 'A'
    Returns empty string if invalid/unknown class.
    """
    if label is None:
        return ""
    clean = str(label).strip()

    # Remove leading numbering like "1. ", "01_", etc.
    clean = re.sub(r"^\d+[\._\-\s]+", "", clean).strip()
    clean = clean.upper()

    word_digit_map = {
        "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
        "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"
    }

    if clean in word_digit_map:
        return word_digit_map[clean]

    if clean in TARGET_36_CLASSES:
        return clean

    return ""


def save_label_map(label_map: Dict[str, int], output_path: Path) -> None:
    """Saves label map dict to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)


def load_label_map(json_path: Path) -> Dict[str, int]:
    """Loads label map dict from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label: str) -> str:
    """Removes dataset prefix numbering and cleans punctuation for user-facing display."""
    if not label:
        return ""
    clean = re.sub(r"^\d+\.\s*", "", str(label)).strip()
    clean = re.sub(r"\s*\(.*?\)", "", clean).strip()
    return clean


def label_to_text(label: str) -> str:
    """Converts a dataset label into a clean, natural English spoken phrase."""
    text = normalize_label(label)
    if not text:
        return "Unknown sign"
    return text[0].upper() + text[1:]
