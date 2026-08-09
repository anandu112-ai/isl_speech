"""
Text-to-Speech (TTS) Engine for INCLUDE-50 Sign-to-Speech System.
Uses pyttsx3 for offline Windows text-to-speech output.
"""

from typing import Optional
import pyttsx3

from src.utils.labels import label_to_text


class SpeechEngine:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.engine: Optional[pyttsx3.Engine] = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"Warning: Could not initialize pyttsx3 TTS engine: {e}")
            self.engine = None

    def speak(self, text_or_label: str):
        """
        Normalizes label and pronounces spoken phrase.
        """
        phrase = label_to_text(text_or_label)
        print(f"[SPEECH OUTPUT] '{phrase}'")

        if self.engine is not None:
            try:
                self.engine.say(phrase)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Speech error: {e}")
