"""
Text-to-Speech CLI Utility Script for INCLUDE-50.
Takes a sign label or text string and pronounces it aloud.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speech.tts import SpeechEngine


def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech Sign Phrase Pronunciation")
    parser.add_argument("--text", type=str, required=True, help="Text phrase or dataset label to speak")
    args = parser.parse_args()

    engine = SpeechEngine()
    engine.speak(args.text)


if __name__ == "__main__":
    main()
