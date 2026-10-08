import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_field(value, max_len: int = 300) -> str:
    """Makes an untrusted value safe to embed inside a tagged data block."""
    text = unicodedata.normalize("NFKC", str(value))  # fullwidth ＜ -> <
    text = _CONTROL_CHARS.sub("", text)               # strip control chars
    text = text.replace("<", "‹").replace(">", "›")   # value can't open/close tags
    text = " ".join(text.split())                     # no newlines: no fake fields/turns
    return text[:max_len]