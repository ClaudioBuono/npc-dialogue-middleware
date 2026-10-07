import logging
import re
from threading import Lock
import time
from core.configuration.settings import Settings
from core.types.contexts import GameContext, NPCContext
from core.types.enums import Language
from core.tools.lexicon_scanner import FastLexiconScanner, StreamingLexiconScanner
from api.schemas import ComposedDialogue, DialogueOptionsSchema, QuestChoiceSchema
from core.tools.lexicon_scanner import StreamingLexiconScanner

logger = logging.getLogger(__name__)


class Guardrail:
    """Provides validation mechanics to scan dialogue outputs."""

    def __init__(self) -> None:
        """Initialize the guardrail, building the lexicon for the current
        language and subscribing to future language changes."""
        self._scanner_lock = Lock()
        self.lexicon_scanner: FastLexiconScanner | None = None
        self._build_lexicon(Settings().language)
        Settings.language_changed.connect(self._on_language_change)

    def _on_language_change(self, sender, language: Language, **kwargs) -> None:
        """Rebuild the lexicon scanner whenever the application language changes.

        Args:
            sender: The object that emitted the language_changed signal.
            language: The new active language.
        """
        logger.info("Language changed to %s", language.name)
        self._build_lexicon(language)

    def _build_lexicon(self, language: Language) -> None:
        """Load the derogatory terms for the given language and replace the
        active lexicon scanner with a new one built from them.

        Args:
            language: The language to build the lexicon for.
        """
        start = time.perf_counter()
        terms = self._load_derogatory_terms(language)
        with self._scanner_lock:
            self.lexicon_scanner = FastLexiconScanner(terms)
        logger.debug("Lexicon built: language=%s, %d terms, %.2fs", language.name, len(terms), time.perf_counter() - start)

    def _retrieve_banned_words(self, text: str) -> list[str]:
        """Return the banned terms found in a string."""
        scanner = self.lexicon_scanner
        return scanner.scan(text)
    
    def retrieve_banned_words_in_composed_dialogue(self, composed_output: ComposedDialogue) -> list[str]:
        """Return the banned terms found in a ComposedDialogue."""
        return self._retrieve_banned_words(self._composed_output_to_text(composed_output))

    def _validate_text(self, text_to_validate: str, source: str) -> bool:
        """Scan a piece of text against the current lexicon for banned words.

        Args:
            text_to_validate: The raw text to scan.
            source: Who is asking to validate the text.

        Returns:
            bool: True if no banned terms were found, False otherwise.
        """
        banned = self._retrieve_banned_words(text_to_validate)
        if banned:
            logger.warning("Lexicon scan of %s: %d banned term(s) found: %s", source, len(banned), banned)
            return False
        logger.debug("Lexicon scan of %s: clean", source)
        return True

    def validate_npc_context(self, npc_context: NPCContext) -> bool:
        """Scan the fields of an NPCContext instance for lexicon violations.

        Args:
            npc_context: The NPC context object whose fields should be validated.

        Returns:
            bool: True if no banned terms were found, False otherwise.
        """
        data = npc_context.model_dump(exclude_none=True)
        text = " ".join(f"{key}: {value}" for key, value in data.items())

        return self._validate_text(text, f"NPC context")

    def validate_game_context(self, game_context: GameContext) -> bool:
        """Scan the fields of a GameContext instance for lexicon violations.

        Args:
            game_context: The game context object whose fields should be validated.

        Returns:
            bool: True if no banned terms were found, False otherwise.
        """
        data = game_context.model_dump(exclude_none=True)
        text = " ".join(f"{key}: {value}" for key, value in data.items())

        return self._validate_text(text, "game context")

    def validate_composed_output(self, composed_output: ComposedDialogue) -> bool:
        """Scan the fields of a ComposedDialogue instance for lexicon violations.

        Args:
            composed_output (ComposedDialogue): The composed dialogue object containing
                text fields and optional dialogue choices to validate.

        Returns:
            bool: True if no banned terms were found, False otherwise.
        """
        return not self.retrieve_banned_words_in_composed_dialogue(composed_output)

    
    def validate_censor_word(self, censor_word: str) -> bool:
        """Scan a candidate censor word for lexicon violations.

        Args:
            censor_word: The word being proposed as the new censor word.

        Returns:
            bool: True if the word contains no banned terms, False otherwise.
        """
        return self._validate_text(censor_word, "censor word")

    def get_streaming_scanner(self) -> StreamingLexiconScanner:
        """Returns a new StreamingLexiconScanner bound to the current lexicon."""
        with self._scanner_lock:
            scanner = self.lexicon_scanner
        return StreamingLexiconScanner(scanner)

    @staticmethod
    def _load_derogatory_terms(language: Language) -> set[str]:
        """Loads and filters derogatory terms from the HurtLex dataset for a given language.

        Args:
            language: the language for which to load the lexicon.

        Returns:
            set[str]: A set of unique lemma strings marked as conservative derogatory terms.

        Raises:
            KeyError: if no HurtLex dataset exists for the requested language.
        """
        
        import pandas as pd
        from core.helpers.paths import resource_path

        hurtlex_filename_by_language = {
            Language.ENGLISH: "hurtlex_EN.tsv",
            Language.ITALIAN: "hurtlex_IT.tsv",
        }

        try:
            filename = hurtlex_filename_by_language[language]
        except KeyError as exc:
            raise KeyError(f"No HurtLex dataset configured for language: {language}") from exc

        hurtlex_df = pd.read_csv(resource_path(filename), sep="\t")
        return set(hurtlex_df["lemma"].dropna().tolist())

    @staticmethod
    def redact_terms(text: str, terms: list[str]) -> str:
        """Replaces banned words with the word chosen in settings."""

        for term in terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            text = pattern.sub(Settings().censor_word, text)
        return text

    @staticmethod
    def _composed_output_to_text(composed_output: ComposedDialogue) -> str:
        """Flatten the scannable fields of a ComposedDialogue into a single string."""

        # Convert list fields to string, keeping existing strings intact
        fields = [
            composed_output.dialogue,
            (
                str(composed_output.player_options)
                if isinstance(composed_output.player_options, list)
                else composed_output.player_options
            ),
        ]
 
        # Combine all valid string fields into a single text payload for scanning
        raw_text = " ".join(val for val in fields if isinstance(val, str))
        return raw_text

    def censor_composed_dialogue(self, composed_dialogue: ComposedDialogue) -> ComposedDialogue:
        """Censor the composed dialogue by redacting any terms that are flagged as derogatory."""

        banned_words: list[str] = self.retrieve_banned_words_in_composed_dialogue(composed_dialogue)
        if banned_words:
            logger.info("Censoring dialogue -- %d banned term(s): %s", len(banned_words), banned_words)

        options = composed_dialogue.player_options

        if isinstance(options, QuestChoiceSchema):
            options = options.model_copy(update={
                "accept": self.redact_terms(options.accept, banned_words),
                "refuse": self.redact_terms(options.refuse, banned_words),
                "dialogue_options": (
                    [self.redact_terms(o, banned_words) for o in options.dialogue_options]
                    if options.dialogue_options is not None
                    else None
                ),
            })
        elif isinstance(options, DialogueOptionsSchema):
            options = options.model_copy(update={
                "dialogue_options": [self.redact_terms(o, banned_words) for o in options.dialogue_options],
            })

        return composed_dialogue.model_copy(update={
            "dialogue": self.redact_terms(composed_dialogue.dialogue, banned_words),
            "player_options": options,
        })