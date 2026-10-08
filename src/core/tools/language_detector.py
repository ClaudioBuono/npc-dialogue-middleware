from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
import fasttext
from core.helpers.paths import resource_path
from core.types.enums import Language

logger = logging.getLogger(__name__)

# Compressed fastText language identification model (176 languages)
MODEL_FILENAME = "lid.176.ftz"

# Game languages mapped to their fastText language codes
_TO_FASTTEXT: dict[Language, str] = {
    Language.ENGLISH: "en",
    Language.ITALIAN: "it",
}

# Below this probability the result is considered "uncertain" and is NOT a mismatch
MIN_CONFIDENCE = 0.6

_checker: LanguageChecker | None = None
_checker_lock = Lock()


class LanguageChecker:
    """Detects the language of short texts using the fastText lid.176 model."""

    def __init__(self) -> None:
        """Load the fastText model from the bundled resources.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model path contains non-ASCII characters,
                which fastText cannot open on some platforms (e.g. Windows).
        """
        model_path = Path(resource_path(MODEL_FILENAME))
        if not model_path.exists():
            raise FileNotFoundError(f"Language model not found: {model_path}")
        if not str(model_path).isascii():
            raise RuntimeError(
                f"Language model path contains non-ASCII characters and fastText cannot open it: {model_path}"
            )
        self._model = fasttext.load_model(str(model_path))

    def detect(self, text: str) -> tuple[str, float] | None:
        """Detect the most likely language of a text.

        Args:
            text: The text to analyze. Newlines are collapsed to spaces,
                since fastText does not accept them.

        Returns:
            A (language_code, probability) tuple, e.g. ("it", 0.98),
            or None if the text is empty or no prediction is available.
        """
        text = " ".join(text.split())  # predict does not accept newlines
        if not text:
            return None
        # model.f.predict avoids model.predict, which can fail with numpy>=2
        # (np.array(..., copy=False)). Returns [(prob, "__label__xx"), ...]
        predictions = self._model.f.predict(text, 1, 0.0, "strict")
        if not predictions:
            return None
        prob, label = predictions[0]
        return label.replace("__label__", ""), float(prob)

    def is_wrong_language(self, text: str, expected: Language) -> bool:
        """Check whether a text is confidently written in the wrong language.

        Uncertain results (low probability, empty text, unmapped language) are
        NOT treated as a mismatch, to avoid regenerating correct dialogues
        because of short or ambiguous text.

        Args:
            text: The text to analyze.
            expected: The language the text is supposed to be written in.

        Returns:
            True only if the detected language differs from the expected one
            with a probability of at least MIN_CONFIDENCE, False otherwise.
        """
        expected_code = _TO_FASTTEXT.get(expected)
        if expected_code is None:
            return False
        result = self.detect(text)
        if result is None:
            return False
        code, prob = result
        if code != expected_code and prob >= MIN_CONFIDENCE:
            logger.debug("Language mismatch: expected=%s, detected=%s (p=%.2f)", expected_code, code, prob)
            return True
        return False


def get_language_checker() -> LanguageChecker:
    """Return the shared LanguageChecker instance, building it on first use.

    The model load is the slow part, so the instance is created once and
    reused. Access is thread-safe.

    Returns:
        The shared LanguageChecker.

    Raises:
        FileNotFoundError: If the model file is missing (first call only).
        RuntimeError: If the model path contains non-ASCII characters (first call only).
    """
    global _checker
    with _checker_lock:
        if _checker is None:
            _checker = LanguageChecker()
        return _checker