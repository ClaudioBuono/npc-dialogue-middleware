import ahocorasick
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class _RawMatch:
    """A raw Aho-Corasick match found inside a text window.

    The match has not been checked against word boundaries yet.

    Attributes:
        start: Index of the first character of the match, relative to the window.
        end: Index of the last character of the match (inclusive), relative
            to the window.
        term: The matched (lowercase) term.
    """

    start: int
    end: int
    term: str


@dataclass(frozen=True, slots=True)
class _PendingMatch:
    """A match that touches the end of the stream so far.

    Its right word boundary cannot be verified until the next chunk arrives
    (or the stream ends), so confirmation is deferred.

    Attributes:
        term: The matched (lowercase) term.
        global_start: Index of the first character of the match, relative to
            the whole stream (not just the current window).
    """

    term: str
    global_start: int


class FastLexiconScanner:
    """Efficient multi-pattern text scanner based on the Aho-Corasick algorithm."""

    def __init__(self, terms: set[str]) -> None:
        """Initializes the automaton with a set of target terms.

        Args:
            terms (set[str]): A set of string terms to search for in target texts.
        """
        self.automaton = ahocorasick.Automaton()
        for term in terms:
            self.automaton.add_word(term.lower(), term.lower())
        self.automaton.make_automaton()
        self.max_len = max((len(t) for t in terms), default=0)

    def scan(self, text: str) -> list[str]:
        """Scans input text for matches bounded by whole-word boundaries.

        Ignores matches embedded inside larger alphanumeric words by verifying
        that adjacent characters are non-alphanumeric or string boundaries.

        Args:
            text (str): The input text string to scan.

        Returns:
            list[str]: A list of isolated whole-word matching terms found in the text.
        """
        text_lower = text.lower()
        matched_terms = []
        text_len = len(text_lower)

        # Unpack end_index and matched term payload
        for end_index, term in self.automaton.iter(text_lower):
            term_len = len(term)
            start_index = end_index - term_len + 1

            # 1. Check character BEFORE match
            char_before_is_alphanumeric = (
                start_index > 0 and text_lower[start_index - 1].isalnum()
            )

            # 2. Check character AFTER match
            char_after_is_alphanumeric = (
                end_index + 1 < text_len and text_lower[end_index + 1].isalnum()
            )

            # Accept term only if it is isolated (not part of a larger word)
            if not char_before_is_alphanumeric and not char_after_is_alphanumeric:
                matched_terms.append(term)

        return matched_terms

class StreamingLexiconScanner:
    """Stateful whole-word lexicon scanner for a single text stream.

    Text is fed in arbitrary chunks. Matches that may continue into the next
    chunk (e.g. "ass" at the end of "ass|ume") are held back as pending and
    resolved when more text arrives or when the stream is flushed.
    """

    def __init__(self, lexicon: FastLexiconScanner) -> None:
        """Creates a scanner bound to an existing lexicon.

        Args:
            lexicon: The lexicon whose automaton is shared (not copied).
        """
        self._automaton = lexicon.automaton
        self._max_len = lexicon.max_len
        self._tail = ""
        self._global_offset = 0
        self._pending: _PendingMatch | None = None

    def _raw_matches(self, text: str) -> list[_RawMatch]:
        """Finds all raw automaton matches in `text`, ignoring word boundaries.

        Args:
            text: The lowercase text window to scan.

        Returns:
            A list of `_RawMatch`, with indices relative to `text`.
        """
        return [
            _RawMatch(start=end - len(term) + 1, end=end, term=term)
            for end, term in self._automaton.iter(text)
        ]

    def feed(self, chunk: str) -> list[str]:
        """Processes the next chunk of the stream.

        Args:
            chunk: The next piece of text.

        Returns:
            The whole-word terms confirmed by this chunk. This includes a
            match left pending by the previous chunk, if the new chunk
            confirms it.
        """
        chunk_lower = chunk.lower()
        confirmed: list[str] = []

        # Resolve last pending match
        if self._pending:
            term = self._pending.term
            self._pending = None
            if not chunk_lower or not chunk_lower[0].isalnum():
                confirmed.append(term)

        window = self._tail + chunk_lower
        tail_len = len(self._tail)
        window_len = len(window)

        for match in self._raw_matches(window):
            if match.end < tail_len:
                continue
            global_start = self._global_offset + match.start
            if match.start == 0 and global_start != 0:
                continue
            if match.start > 0 and window[match.start - 1].isalnum():
                continue
            if match.end == window_len - 1:
                self._pending = _PendingMatch(term=match.term, global_start=global_start)
                continue
            if not window[match.end + 1].isalnum():
                confirmed.append(match.term)

        keep = self._max_len
        drop = max(0, window_len - keep)
        self._global_offset += drop
        self._tail = window[-keep:] if keep else ""
        return confirmed

    def flush(self) -> list[str]:
        """Signals the end of the stream and resolves any pending match.

        At the end of the stream there is no following character, so a
        pending match is necessarily a whole word.

        Returns:
            A list with the pending term, or an empty list if none.
        """
        if self._pending:
            term = self._pending.term
            self._pending = None
            return [term]
        return []