"""Unit tests for core.tools.language_detector.find_foreign_words.

The function uses the ``wordfreq`` library to identify words that are much
more common in another supported language than in the expected one.

All tests mock ``zipf_frequency`` so the suite:
- requires no network access,
- is deterministic regardless of library version,
- explicitly documents the detection logic under test.
"""
import pytest
from unittest.mock import patch

from core.tools.language_detector import (
    MIN_ZIPF_GAP,
    OTHER_MIN_ZIPF,
    find_foreign_words,
)
from core.types.enums import Language


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_zipf(mapping: dict):
    """
    Return a side_effect callable for ``zipf_frequency``.

    ``mapping`` maps (word, lang_code) -> frequency.
    Any pair not in the mapping defaults to 0.0.
    """
    def _side_effect(word, lang):
        return mapping.get((word, lang), 0.0)
    return _side_effect


# ---------------------------------------------------------------------------
# Basic happy paths
# ---------------------------------------------------------------------------

class TestFindForeignWordsHappyPaths:

    def test_returns_list(self):
        with patch("core.tools.language_detector.zipf_frequency", return_value=0.0):
            result = find_foreign_words("hello world", Language.ENGLISH)
        assert isinstance(result, list)

    def test_empty_text_returns_empty_list(self):
        with patch("core.tools.language_detector.zipf_frequency", return_value=0.0):
            result = find_foreign_words("", Language.ENGLISH)
        assert result == []

    def test_text_with_no_alphabetic_words_returns_empty(self):
        """Numbers, punctuation, and short tokens (< 3 chars) are skipped."""
        with patch("core.tools.language_detector.zipf_frequency", return_value=0.0):
            result = find_foreign_words("123 !! -- ok", Language.ENGLISH)
        assert result == []

    def test_words_shorter_than_three_chars_are_ignored(self):
        """The regex only matches tokens of 3+ letters."""
        with patch("core.tools.language_detector.zipf_frequency", return_value=0.0):
            result = find_foreign_words("I am ok", Language.ENGLISH)
        assert result == []

    def test_no_foreign_words_all_below_threshold(self):
        """No word is flagged when the other-language frequency is below OTHER_MIN_ZIPF."""
        freq_map = {
            ("hello", "en"): 6.0,
            ("hello", "it"): 1.0,  # below OTHER_MIN_ZIPF (3.0)
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("hello", Language.ENGLISH)
        assert result == []

    def test_no_foreign_words_gap_too_small(self):
        """Frequency gap must be >= MIN_ZIPF_GAP; a smaller gap means the word is not flagged."""
        # other=3.5, expected=2.5 -> gap=1.0 < MIN_ZIPF_GAP (1.5)
        freq_map = {
            ("ciao", "en"): 2.5,
            ("ciao", "it"): 3.5,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("ciao", Language.ENGLISH)
        assert result == []


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

class TestForeignWordDetection:

    def test_single_foreign_word_detected(self):
        """A word common in Italian but rare in English is flagged when processing English text."""
        freq_map = {
            ("persona", "en"): 2.0,
            ("persona", "it"): 6.0,  # >= OTHER_MIN_ZIPF and gap = 4.0 >= MIN_ZIPF_GAP
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("A persona appeared.", Language.ENGLISH)
        assert "persona" in result

    def test_multiple_foreign_words_detected(self):
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
            ("cosa", "en"): 0.5,
            ("cosa", "it"): 5.5,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("La persona e la cosa.", Language.ENGLISH)
        assert "persona" in result
        assert "cosa" in result

    def test_order_of_appearance_preserved(self):
        """Words returned in the order they first appear in the text."""
        freq_map = {
            ("alpha", "en"): 0.5,
            ("alpha", "it"): 5.0,
            ("beta", "en"): 0.5,
            ("beta", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("alpha and then beta", Language.ENGLISH)
        assert result == ["alpha", "beta"]

    def test_no_duplicates_when_word_appears_twice(self):
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("persona and another persona", Language.ENGLISH)
        assert result.count("persona") == 1

    def test_case_insensitive_detection(self):
        """The scanner lowercases words before lookup; 'Persona' and 'PERSONA' are the same."""
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("The PERSONA is here.", Language.ENGLISH)
        assert result == ["persona"]

    def test_mixed_language_direction_italian_expected(self):
        """When Italian is expected, an English-dominant word is flagged."""
        freq_map = {
            ("the", "it"): 0.0,  # short – will be filtered by regex anyway
            ("house", "en"): 6.0,
            ("house", "it"): 1.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("The house was big.", Language.ITALIAN)
        assert "house" in result

    def test_boundary_exactly_at_threshold(self):
        """A word precisely at OTHER_MIN_ZIPF with exactly MIN_ZIPF_GAP should be flagged."""
        freq_map = {
            ("word", "en"): OTHER_MIN_ZIPF - MIN_ZIPF_GAP,
            ("word", "it"): OTHER_MIN_ZIPF,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("word", Language.ENGLISH)
        assert result == ["word"]

    def test_boundary_just_below_threshold_not_flagged(self):
        """A gap of MIN_ZIPF_GAP - epsilon should NOT trigger flagging."""
        epsilon = 0.01
        freq_map = {
            ("word", "en"): OTHER_MIN_ZIPF - MIN_ZIPF_GAP + epsilon,
            ("word", "it"): OTHER_MIN_ZIPF,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("word", Language.ENGLISH)
        assert result == []


# ---------------------------------------------------------------------------
# Ignore set
# ---------------------------------------------------------------------------

class TestIgnoreSet:

    def test_ignored_word_not_reported(self):
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("persona here", Language.ENGLISH, ignore={"persona"})
        assert "persona" not in result

    def test_ignore_is_case_agnostic(self):
        """Words are lowercased before checking the ignore set."""
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("PERSONA here", Language.ENGLISH, ignore={"persona"})
        assert "persona" not in result

    def test_ignore_none_defaults_to_empty_set(self):
        """Passing ignore=None is equivalent to passing an empty ignore set."""
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result_none = find_foreign_words("persona", Language.ENGLISH, ignore=None)
            result_empty = find_foreign_words("persona", Language.ENGLISH, ignore=set())
        assert result_none == result_empty

    def test_ignored_word_does_not_block_others(self):
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
            ("cosa", "en"): 0.5,
            ("cosa", "it"): 5.5,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("persona and cosa", Language.ENGLISH, ignore={"persona"})
        assert "persona" not in result
        assert "cosa" in result

    def test_empty_ignore_set_reports_all_foreign(self):
        freq_map = {
            ("persona", "en"): 1.0,
            ("persona", "it"): 5.0,
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("persona", Language.ENGLISH, ignore=set())
        assert "persona" in result


# ---------------------------------------------------------------------------
# Edge cases – text variety
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_only_punctuation_no_words_found(self):
        with patch("core.tools.language_detector.zipf_frequency", return_value=5.0):
            result = find_foreign_words("!!! ??? --- ...", Language.ENGLISH)
        assert result == []

    def test_digits_mixed_with_letters_not_matched(self):
        """Tokens containing digits are not matched (regex: \\W\\d_ excluded)."""
        with patch("core.tools.language_detector.zipf_frequency", return_value=5.0):
            result = find_foreign_words("abc123 and 999xyz", Language.ENGLISH)
        # abc123 and 999xyz are not purely alphabetic
        assert "abc123" not in result
        assert "999xyz" not in result

    def test_unknown_word_in_all_languages_not_flagged(self):
        """Fantasy names are unknown in every language so they should not be flagged."""
        with patch("core.tools.language_detector.zipf_frequency", return_value=0.0):
            result = find_foreign_words("xyranthul visited", Language.ENGLISH)
        assert "xyranthul" not in result

    def test_shared_loanword_not_flagged(self):
        """Words like 'pizza' appear frequently in both languages: gap is small."""
        freq_map = {
            ("pizza", "en"): 4.5,
            ("pizza", "it"): 5.5,  # gap = 1.0 < MIN_ZIPF_GAP
        }
        with patch("core.tools.language_detector.zipf_frequency", side_effect=_mock_zipf(freq_map)):
            result = find_foreign_words("I love pizza", Language.ENGLISH)
        assert "pizza" not in result
