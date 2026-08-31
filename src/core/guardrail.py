import logging
import time
from tools.hurtlex_loader import load_derogatory_terms
from tools.lexicon_scanner import FastLexiconScanner
from core.types.dataclasses import ComposedDialogue

logger = logging.getLogger(__name__)


class Guardrail:
    """Provides validation mechanics to scan dialogue outputs"""

    def __init__(self) -> None:
        self.lexicon_scanner = FastLexiconScanner(load_derogatory_terms())

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
                str(composed_output.dialogue_options)
                if isinstance(composed_output.dialogue_options, list)
                else composed_output.dialogue_options
            ),
        ]

        # Combine all valid string fields into a single text payload for scanning
        raw_Text = " ".join(val for val in fields if isinstance(val, str))
        
        # Scan for banned words 
        start_time = time.perf_counter()
        scan_result: list[str] = self.lexicon_scanner.scan(raw_Text)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Output scanned in {execution_time_ms}ms.")
        logger.info(f"Fairness scan '{scan_result}'")

        return len(scan_result) == 0