import re
from wordfreq import zipf_frequency

from core.types.enums import Language

_CODES = {Language.ENGLISH: "en", Language.ITALIAN: "it"}
_WORD_RE = re.compile(r"[^\W\d_]{3,}")   # letters only, at least 3 characters

OTHER_MIN_ZIPF = 3.0   # common enough in the other language
MIN_ZIPF_GAP = 1.5     # and at least this many Zipf points above the expected language


def find_foreign_words(text: str, expected: Language, ignore: set[str] | None = None) -> list[str]:
    """Return the words that are much more frequent in another game language than in the expected one.

    The comparison is relative, not absolute: a word is reported when it is
    common in another language (zipf >= OTHER_MIN_ZIPF) and at least
    MIN_ZIPF_GAP points more frequent there than in the expected language.
    This catches words that also exist in the expected language's corpus as
    quoted foreign words (e.g. "persone", "provincia"), while shared words
    and loanwords ("pizza", "come") are not reported.

    A single occurrence is enough to be reported. Words unknown in every
    language (fantasy names) and words listed in `ignore` are skipped.

    Args:
        text: The text to analyze.
        expected: The language the text is supposed to be written in.
        ignore: Lowercase words to skip (e.g. the NPC's name, places).

    Returns:
        The foreign words found, in order of appearance, without duplicates.
    """
    expected_code = _CODES[expected]
    others = [c for c in _CODES.values() if c != expected_code]
    ignore = ignore or set()

    found: list[str] = []
    for word in _WORD_RE.findall(text.lower()):
        if word in ignore or word in found:
            continue
        f_expected = zipf_frequency(word, expected_code)
        if any(
            (f_other := zipf_frequency(word, code)) >= OTHER_MIN_ZIPF
            and f_other - f_expected >= MIN_ZIPF_GAP
            for code in others
        ):
            found.append(word)
    return found