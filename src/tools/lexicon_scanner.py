import ahocorasick


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
    """Scan state for a single streaming."""

    def __init__(self, lexicon: 'FastLexiconScanner') -> None:
        self._automaton = lexicon.automaton
        self._max_len = lexicon.max_len
        self._tail = ""
        self._global_offset = 0
        self._pending: tuple[str, int] | None = None

    def _raw_matches(self, text: str) -> list[tuple[int, int, str]]:
        return [
            (end - len(term) + 1, end, term)
            for end, term in self._automaton.iter(text)
        ]

    def feed(self, chunk: str) -> list[str]:
        window = self._tail + chunk.lower()
        tail_len = len(self._tail)
        window_len = len(window)
        confirmed: list[str] = []

        for start, end, term in self._raw_matches(window):
            if end < tail_len:
                continue
            global_start = self._global_offset + start
            if start == 0 and global_start != 0:
                continue
            if start > 0 and window[start - 1].isalnum():
                continue
            if end == window_len - 1:
                self._pending = (term, global_start)
                continue
            if not window[end + 1].isalnum():
                confirmed.append(term)

        keep = self._max_len
        drop = max(0, window_len - keep)
        self._global_offset += drop
        self._tail = window[-keep:] if keep else ""
        return confirmed

    def flush(self) -> list[str]:
        if self._pending:
            term, _ = self._pending
            self._pending = None
            return [term]
        return []