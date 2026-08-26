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

    def scan_advanced(self, text: str) -> list[str]:
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