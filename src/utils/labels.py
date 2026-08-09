"""
Utility functions for text normalization, dataset label parsing, and speech text formatting.
"""

import re


def normalize_label(label: str) -> str:
    """
    Removes dataset prefix numbering and cleans punctuation for user-facing display.
    Examples:
    "1. Dog" -> "Dog"
    "51. Good Morning" -> "Good Morning"
    "46. you (plural)" -> "you"
    "28. Store or Shop" -> "Store or Shop"
    """
    if not label:
        return ""

    # Remove leading numbering like "1. ", "51. ", "104. "
    clean = re.sub(r"^\d+\.\s*", "", str(label)).strip()

    # Remove trailing parenthetical modifiers like " (plural)"
    clean = re.sub(r"\s*\(.*?\)", "", clean).strip()

    return clean


def label_to_text(label: str) -> str:
    """
    Converts a dataset label into a clean, natural English spoken phrase.
    """
    text = normalize_label(label)
    if not text:
        return "Unknown sign"

    # Make first letter uppercase, rest standard casing
    return text[0].upper() + text[1:]
