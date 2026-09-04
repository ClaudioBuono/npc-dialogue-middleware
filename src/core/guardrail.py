import logging
from threading import Lock
import time
from core.config.settings import Settings
from core.types.enums import Language
from tools.lexicon_scanner import FastLexiconScanner, StreamingLexiconScanner
from api.schemas import ComposedDialogue
from tools.lexicon_scanner import StreamingLexiconScanner

logger = logging.getLogger(__name__)


class Guardrail:
    """Provides validation mechanics to scan dialogue outputs.

    Il lexicon interno viene ricostruito automaticamente ogni volta che
    Settings.change_language() viene chiamato altrove nell'app.
    """

    def __init__(self) -> None:
        self._scanner_lock = Lock()
        self.lexicon_scanner: FastLexiconScanner | None = None
        self._build_lexicon(Settings().language)
        Settings.language_changed.connect(self._on_language_change)

    def _on_language_change(self, sender, language: Language, **kwargs) -> None:
        self._build_lexicon(language)

    def _build_lexicon(self, language: Language) -> None:
        terms = self._load_derogatory_terms(language)
        with self._scanner_lock:
            self.lexicon_scanner = FastLexiconScanner(terms)
        logger.info(f"Lexicon rebuilt for language {language}")

    def validate(self, composed_output: ComposedDialogue) -> bool:
        """Scans the fields of a ComposedDialogue instance for lexicon violations.

        Args:
            composed_output (ComposedDialogue): The composed dialogue object containing
                text fields and optional dialogue choices to validate.
        """

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
        raw_Text = " ".join(val for val in fields if isinstance(val, str))

        # Scan for banned words
        start_time = time.perf_counter()
        with self._scanner_lock:
            scanner = self.lexicon_scanner
        scan_result: list[str] = scanner.scan(raw_Text)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Output scanned in {execution_time_ms}ms.")
        logger.info(f"Fairness scan '{scan_result}'")

        return len(scan_result) == 0

    def get_streaming_scanner(self) -> StreamingLexiconScanner:
        """Returns a new StreamingLexiconScanner bound to the current lexicon."""
        with self._scanner_lock:
            scanner = self.lexicon_scanner
        return StreamingLexiconScanner(scanner)

    @staticmethod
    def _load_derogatory_terms(language: Language) -> set[str]:
        """Loads and filters derogatory terms from the HurtLex dataset for a given language.

        Args:
            language: la lingua per cui caricare il lessico.

        Returns:
            set[str]: A set of unique lemma strings marked as conservative derogatory terms.

        Raises:
            KeyError: se non esiste un dataset HurtLex per la lingua richiesta.
        """
        import pandas as pd
        from core.paths import resource_path

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