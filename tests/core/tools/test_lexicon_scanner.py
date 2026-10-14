"""Unit tests for core.tools.lexicon_scanner.

Covers:
- FastLexiconScanner: construction, whole-word scan, edge cases
- StreamingLexiconScanner: chunk feeding, pending match resolution, flush
"""
import pytest

from core.tools.lexicon_scanner import FastLexiconScanner, StreamingLexiconScanner


# ---------------------------------------------------------------------------
# FastLexiconScanner
# ---------------------------------------------------------------------------

class TestFastLexiconScannerInit:

    def test_empty_terms_set_constructs_without_error(self):
        scanner = FastLexiconScanner(set())
        assert scanner.max_len == 0

    def test_single_term_max_len(self):
        scanner = FastLexiconScanner({"hello"})
        assert scanner.max_len == 5

    def test_multiple_terms_max_len(self):
        scanner = FastLexiconScanner({"hi", "hello", "hey"})
        assert scanner.max_len == 5  # len("hello")

    def test_terms_stored_lowercase(self):
        """The automaton should match the lowercased version of the term."""
        scanner = FastLexiconScanner({"HELLO"})
        result = scanner.scan("HELLO world")
        assert "hello" in result


class TestFastLexiconScannerScan:

    # --- Happy paths ---

    def test_exact_match_isolated_word(self):
        scanner = FastLexiconScanner({"hate"})
        assert scanner.scan("I hate this") == ["hate"]

    def test_multiple_distinct_matches(self):
        scanner = FastLexiconScanner({"kill", "hate"})
        result = scanner.scan("I hate and kill")
        assert "hate" in result
        assert "kill" in result

    def test_match_at_start_of_string(self):
        scanner = FastLexiconScanner({"kill"})
        assert scanner.scan("kill him") == ["kill"]

    def test_match_at_end_of_string(self):
        scanner = FastLexiconScanner({"kill"})
        assert scanner.scan("he will kill") == ["kill"]

    def test_match_surrounded_by_punctuation(self):
        scanner = FastLexiconScanner({"kill"})
        result = scanner.scan("...kill!")
        assert "kill" in result

    def test_case_insensitive_scan(self):
        scanner = FastLexiconScanner({"hate"})
        result = scanner.scan("I HATE this")
        assert "hate" in result

    def test_single_word_text_that_matches(self):
        scanner = FastLexiconScanner({"kill"})
        assert scanner.scan("kill") == ["kill"]

    # --- Whole-word boundary enforcement ---

    def test_embedded_term_not_matched_within_larger_word(self):
        """'ass' inside 'class' must NOT be flagged."""
        scanner = FastLexiconScanner({"ass"})
        assert scanner.scan("The class was good") == []

    def test_term_embedded_at_start_of_word_not_matched(self):
        """'kill' in 'killer' → no match."""
        scanner = FastLexiconScanner({"kill"})
        assert scanner.scan("he is a killer") == []

    def test_term_embedded_at_end_of_word_not_matched(self):
        """'ill' in 'will' → no match."""
        scanner = FastLexiconScanner({"ill"})
        assert scanner.scan("he will stay") == []

    def test_partial_substring_not_matched(self):
        scanner = FastLexiconScanner({"hate"})
        assert scanner.scan("whatever") == []

    # --- No matches ---

    def test_empty_text_returns_empty(self):
        scanner = FastLexiconScanner({"hate"})
        assert scanner.scan("") == []

    def test_no_terms_in_text(self):
        scanner = FastLexiconScanner({"kill"})
        assert scanner.scan("everything is fine") == []

    def test_empty_lexicon_raises_on_scan(self):
        """ahocorasick requires at least one word; iterating an empty automaton raises."""
        scanner = FastLexiconScanner(set())
        with pytest.raises(AttributeError):
            scanner.scan("kill hate")

    # --- Repeated matches ---

    def test_same_term_appearing_twice_reported_twice(self):
        scanner = FastLexiconScanner({"hate"})
        result = scanner.scan("I hate or hate you")
        assert result.count("hate") == 2

    # --- Separator edge cases ---

    def test_match_adjacent_to_comma(self):
        scanner = FastLexiconScanner({"hate"})
        assert scanner.scan("I hate, you know") == ["hate"]

    def test_match_adjacent_to_newline(self):
        scanner = FastLexiconScanner({"hate"})
        assert scanner.scan("I\nhate\nthings") == ["hate"]

    def test_match_with_underscore_adjacent_is_alphanumeric_boundary(self):
        """Underscore is NOT alphanumeric per Python's isalnum(); 'hate_' is detected."""
        scanner = FastLexiconScanner({"hate"})
        result = scanner.scan("hate_team")
        # underscore at position after 'e' is not alnum → match is accepted
        assert "hate" in result


# ---------------------------------------------------------------------------
# StreamingLexiconScanner
# ---------------------------------------------------------------------------

class TestStreamingLexiconScannerInit:

    def test_constructs_without_error(self):
        lexicon = FastLexiconScanner({"hate"})
        streaming = StreamingLexiconScanner(lexicon)
        assert streaming is not None

    def test_initial_state_is_clean(self):
        lexicon = FastLexiconScanner({"hate"})
        streaming = StreamingLexiconScanner(lexicon)
        assert streaming._tail == ""
        assert streaming._global_offset == 0
        assert streaming._pending is None


class TestStreamingLexiconScannerFeed:

    def _scanner(self, terms: set[str]) -> StreamingLexiconScanner:
        return StreamingLexiconScanner(FastLexiconScanner(terms))

    # --- Basic single-chunk ---

    def test_single_chunk_clear_match(self):
        s = self._scanner({"hate"})
        result = s.feed("I hate this")
        assert "hate" in result

    def test_single_chunk_no_match(self):
        s = self._scanner({"hate"})
        result = s.feed("everything is fine")
        assert result == []

    def test_empty_chunk_returns_empty(self):
        s = self._scanner({"hate"})
        assert s.feed("") == []

    # --- Multi-chunk: term fully in one chunk ---

    def test_multiple_chunks_term_in_second_chunk(self):
        s = self._scanner({"kill"})
        first = s.feed("some text ")
        second = s.feed("will kill him")
        assert "kill" in second
        assert "kill" not in first

    def test_multiple_chunks_term_in_first_chunk(self):
        s = self._scanner({"hate"})
        first = s.feed("I hate ")
        assert "hate" in first

    # --- Cross-chunk boundary resolution ---

    def test_term_split_across_chunks_detected_on_second(self):
        """'hate' split as 'ha' | 'te more' must be detected when the second chunk arrives."""
        s = self._scanner({"hate"})
        first = s.feed("ha")  # partial – no confirmed match yet
        second = s.feed("te more")  # 'hate' should now be in the window
        combined = first + second
        assert "hate" in combined

    def test_pending_match_resolved_to_confirmed_when_boundary_follows(self):
        """When a chunk ends exactly on the last character of a term, confirmation is deferred."""
        s = self._scanner({"hate"})
        # Feed "hate" alone – ends at boundary → pending
        first = s.feed("hate")
        # Next chunk starts with a space (non-alnum) → word boundary confirmed
        second = s.feed(" things")
        assert "hate" in first + second

    def test_pending_match_cancelled_when_alnum_follows(self):
        """A pending match is discarded when the next chunk begins with an alnum character."""
        s = self._scanner({"ass"})
        # 'ass' at the end of the chunk – is it 'ass' alone or 'assume'?
        first = s.feed("ass")
        # Next chunk starts with 'ume' → the full word is 'assume' → cancel pending
        second = s.feed("ume is good")
        all_found = first + second
        assert "ass" not in all_found

    # --- Case insensitivity ---

    def test_case_insensitive_across_chunks(self):
        s = self._scanner({"hate"})
        first = s.feed("I HATE ")
        second = s.feed("this")
        assert "hate" in first + second

    # --- Multiple matches across stream ---

    def test_multiple_matches_across_multiple_chunks(self):
        s = self._scanner({"kill", "hate"})
        r1 = s.feed("I hate ")
        r2 = s.feed("to kill ")
        r3 = s.feed("things")
        all_found = r1 + r2 + r3
        assert "hate" in all_found
        assert "kill" in all_found

    # --- Same term repeated ---

    def test_same_term_repeated_in_stream(self):
        s = self._scanner({"hate"})
        r1 = s.feed("hate ")
        r2 = s.feed("hate again ")
        assert r1.count("hate") >= 1
        assert r2.count("hate") >= 1

    # --- Empty lexicon ---

    def test_empty_lexicon_raises_on_feed(self):
        """ahocorasick raises when iterating an automaton with no words added."""
        s = StreamingLexiconScanner(FastLexiconScanner(set()))
        with pytest.raises(AttributeError):
            s.feed("kill hate slur")


class TestStreamingLexiconScannerFlush:

    def _scanner(self, terms: set[str]) -> StreamingLexiconScanner:
        return StreamingLexiconScanner(FastLexiconScanner(terms))

    def test_flush_with_no_pending_returns_empty(self):
        s = self._scanner({"hate"})
        s.feed("everything is fine ")
        assert s.flush() == []

    def test_flush_resolves_trailing_pending_match(self):
        """If stream ends immediately after the term, flush must confirm it."""
        s = self._scanner({"hate"})
        s.feed("I hate")  # 'hate' at end of stream: pending
        result = s.flush()
        # The pending match is at the end of the stream → whole word → confirmed
        assert "hate" in result

    def test_flush_clears_pending_state(self):
        s = self._scanner({"hate"})
        s.feed("I hate")
        s.flush()
        # A second flush on the now-clean scanner returns empty
        assert s.flush() == []

    def test_flush_on_fresh_scanner_returns_empty(self):
        s = self._scanner({"hate"})
        assert s.flush() == []

    def test_full_stream_explicit_end(self):
        """Simulate a complete streaming dialogue with flush at the end."""
        s = self._scanner({"kill", "hate"})
        chunks = ["I hate ", "to kill"]
        results = [s.feed(c) for c in chunks]
        results.append(s.flush())
        all_found = [term for r in results for term in r]
        assert "hate" in all_found
        assert "kill" in all_found


# ---------------------------------------------------------------------------
# Integration: FastLexiconScanner shared by multiple StreamingLexiconScanners
# ---------------------------------------------------------------------------

class TestSharedLexicon:
    """Multiple streaming scanners can share the same FastLexiconScanner."""

    def test_two_independent_streaming_scanners(self):
        lexicon = FastLexiconScanner({"hate"})
        s1 = StreamingLexiconScanner(lexicon)
        s2 = StreamingLexiconScanner(lexicon)

        # Feed different text to each
        r1 = s1.feed("I hate this")
        r2 = s2.feed("everything is fine")

        assert "hate" in r1
        assert "hate" not in r2

    def test_shared_lexicon_state_not_corrupted(self):
        """Processing one scanner should not affect another's pending state."""
        lexicon = FastLexiconScanner({"hate"})
        s1 = StreamingLexiconScanner(lexicon)
        s2 = StreamingLexiconScanner(lexicon)

        s1.feed("I hate")  # s1 has a pending match
        # s2 should remain clean
        assert s2._pending is None
