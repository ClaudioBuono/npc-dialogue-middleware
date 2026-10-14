import threading
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from api.schemas import ComposedDialogue, DialogueOptionsSchema, QuestChoiceSchema
from core.configuration.settings import Settings
from core.pipeline.guardrail import Guardrail
from core.tools.lexicon_scanner import FastLexiconScanner, StreamingLexiconScanner
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.enums import Language


@pytest.fixture
def mock_load_terms():
    with patch.object(
        Guardrail, "_load_derogatory_terms", return_value={"badword", "curse", "profane"}
    ) as mock_load:
        yield mock_load


@pytest.fixture
def guardrail(mock_load_terms):
    return Guardrail()


@pytest.fixture
def clean_npc_context():
    return NPCContext(
        name="Elidor",
        age=28,
        personality="Helpful and courteous",
        context="Royal Library",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Allied scholar",
        intent=Dialogue(type="Dialogue", has_options=False),
    )


@pytest.fixture
def clean_game_context():
    return GameContext(
        environment="Citadel",
        epoch="Golden Era",
        world_state="Prosperous",
        main_character_description="An honorable knight",
    )


# --- Initialization & Language Lifecycle Tests ---

def test_guardrail_initialization(guardrail, mock_load_terms):
    """Guardrail loads terms for the current language and initializes the lexicon scanner."""
    mock_load_terms.assert_called_once_with(Settings().language)
    assert guardrail.lexicon_scanner is not None
    assert isinstance(guardrail.lexicon_scanner, FastLexiconScanner)


def test_guardrail_on_language_change_rebuilds_scanner(guardrail, mock_load_terms):
    """_on_language_change triggers _build_lexicon with the newly provided language."""
    guardrail._on_language_change(sender=None, language=Language.ITALIAN)
    mock_load_terms.assert_called_with(Language.ITALIAN)
    assert guardrail.lexicon_scanner is not None


def test_guardrail_language_changed_signal_integration(mock_load_terms):
    """Emitting Settings.language_changed signal rebuilds the active Guardrail lexicon."""
    g = Guardrail()
    mock_load_terms.reset_mock()

    Settings.language_changed.send(Settings, language=Language.ITALIAN)
    mock_load_terms.assert_called_with(Language.ITALIAN)


def test_load_derogatory_terms_english():
    """_load_derogatory_terms loads English HurtLex TSV and returns lemmas set."""
    dummy_df = pd.DataFrame({"lemma": ["badword", "curse", None]})
    with patch("pandas.read_csv", return_value=dummy_df) as mock_read_csv, \
         patch("core.helpers.paths.resource_path", return_value="dummy/hurtlex_EN.tsv") as mock_path:
        terms = Guardrail._load_derogatory_terms(Language.ENGLISH)
        mock_path.assert_called_once_with("hurtlex_EN.tsv")
        mock_read_csv.assert_called_once_with("dummy/hurtlex_EN.tsv", sep="\t")
        assert terms == {"badword", "curse"}


def test_load_derogatory_terms_italian():
    """_load_derogatory_terms loads Italian HurtLex TSV and returns lemmas set."""
    dummy_df = pd.DataFrame({"lemma": ["insulto", "parolaccia"]})
    with patch("pandas.read_csv", return_value=dummy_df) as mock_read_csv, \
         patch("core.helpers.paths.resource_path", return_value="dummy/hurtlex_IT.tsv") as mock_path:
        terms = Guardrail._load_derogatory_terms(Language.ITALIAN)
        mock_path.assert_called_once_with("hurtlex_IT.tsv")
        assert terms == {"insulto", "parolaccia"}


def test_load_derogatory_terms_unsupported_language_raises_key_error():
    """Requesting terms for an unconfigured language raises KeyError."""
    class FakeLanguage:
        name = "KOBOLD"

    with pytest.raises(KeyError, match="No HurtLex dataset configured"):
        Guardrail._load_derogatory_terms(FakeLanguage)


# --- Validation Methods Tests ---

def test_validate_npc_context_clean(guardrail, clean_npc_context):
    """validate_npc_context returns True when all fields contain clean language."""
    assert guardrail.validate_npc_context(clean_npc_context) is True


@pytest.mark.parametrize("dirty_field,dirty_val", [
    ("name", "Bob badword"),
    ("personality", "Highly profane attitude"),
    ("context", "In a curse dungeon"),
    ("main_character_relation", "Harbors badword hatred"),
])
def test_validate_npc_context_dirty(guardrail, clean_npc_context, dirty_field, dirty_val):
    """validate_npc_context returns False when any NPC context field contains banned terms."""
    dirty_npc = clean_npc_context.model_copy(update={dirty_field: dirty_val})
    assert guardrail.validate_npc_context(dirty_npc) is False
    

def test_validate_game_context_clean(guardrail, clean_game_context):
    """validate_game_context returns True when context has no banned words."""
    assert guardrail.validate_game_context(clean_game_context) is True


@pytest.mark.parametrize("dirty_field,dirty_val", [
    ("environment", "Dark curse cave"),
    ("epoch", "Era of badword strife"),
    ("world_state", "Full of profane turmoil"),
    ("main_character_description", "A badword renegade"),
])
def test_validate_game_context_dirty(guardrail, clean_game_context, dirty_field, dirty_val):
    """validate_game_context returns False when any field contains banned words."""
    dirty_game = clean_game_context.model_copy(update={dirty_field: dirty_val})
    assert guardrail.validate_game_context(dirty_game) is False


def test_validate_composed_output_clean(guardrail):
    """validate_composed_output returns True for clean dialogues."""
    output = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="Welcome to our kingdom, brave adventurer.",
    )
    assert guardrail.validate_composed_output(output) is True


def test_validate_composed_output_dirty(guardrail):
    """validate_composed_output returns False when dialogue text contains banned words."""
    output = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="You are a foul badword and deserve no respect.",
    )
    assert guardrail.validate_composed_output(output) is False


def test_validate_censor_word_clean(guardrail):
    """validate_censor_word returns True for benign censor replacement placeholders."""
    assert guardrail.validate_censor_word("[BLEEP]") is True
    assert guardrail.validate_censor_word("****") is True


def test_validate_censor_word_dirty(guardrail):
    """validate_censor_word returns False if proposed censor word itself contains derogatory term."""
    assert guardrail.validate_censor_word("[badword]") is False


# --- Redaction & Regex Tests ---

def test_redact_terms_single_term(guardrail):
    """redact_terms replaces an identified banned term with Settings().censor_word."""
    censor = Settings().censor_word
    result = Guardrail.redact_terms("You are a badword today", ["badword"])
    assert result == f"You are a {censor} today"


def test_redact_terms_multiple_terms(guardrail):
    """redact_terms handles multiple terms across the sentence."""
    censor = Settings().censor_word
    result = Guardrail.redact_terms("Stop that badword and this curse right now", ["badword", "curse"])
    assert result == f"Stop that {censor} and this {censor} right now"


def test_redact_terms_case_insensitive(guardrail):
    """redact_terms is case-insensitive (e.g. BADWORD, Badword, badword)."""
    censor = Settings().censor_word
    result = Guardrail.redact_terms("BADWORD and BadWord and badword", ["badword"])
    assert result == f"{censor} and {censor} and {censor}"


def test_redact_terms_word_boundary_preservation(guardrail):
    """redact_terms only redacts full whole-word matches, not embedded substrings."""
    result = Guardrail.redact_terms("superbadword and badwordcase and curse", ["badword"])
    # "badword" is not replaced inside "superbadword" or "badwordcase" due to \b boundary
    assert "superbadword" in result
    assert "badwordcase" in result


def test_redact_terms_escapes_regex_special_characters():
    """redact_terms safely escapes regex special characters in candidate terms."""
    censor = Settings().censor_word
    special_term = "bad*word?"
    result = Guardrail.redact_terms("Watch your bad*word? here", [special_term])
    assert result == f"Watch your {censor} here"


def test_redact_terms_empty_terms():
    """redact_terms returns original text when terms list is empty."""
    text = "Nothing should change in this sentence."
    assert Guardrail.redact_terms(text, []) == text


def test_redact_terms_empty_text():
    """redact_terms returns empty string when provided empty string."""
    assert Guardrail.redact_terms("", ["badword"]) == ""


# --- censor_composed_dialogue Tests ---

def test_censor_composed_dialogue_with_quest_choice_schema(guardrail):
    """censor_composed_dialogue redacts dialogue, accept, refuse, and quest dialogue_options."""
    censor = Settings().censor_word
    dialogue = ComposedDialogue(
        intent=Quest(
            type="Quest",
            objective="Deliver the relic",
            has_choice=True,
        ),
        dialogue="This is a badword situation.",
        player_options=QuestChoiceSchema(
            accept="I will fix this badword problem.",
            refuse="Deal with your own curse issue.",
            dialogue_options=["Tell me about this profane curse.", "Clean option."],
        ),
    )

    censored = guardrail.censor_composed_dialogue(dialogue)

    assert censored.dialogue == f"This is a {censor} situation."
    assert isinstance(censored.player_options, QuestChoiceSchema)
    assert censored.player_options.accept == f"I will fix this {censor} problem."
    assert censored.player_options.refuse == f"Deal with your own {censor} issue."
    assert censored.player_options.dialogue_options == [
        f"Tell me about this {censor} {censor}.",
        "Clean option.",
    ]


def test_censor_composed_dialogue_with_quest_choice_schema_none_options(guardrail):
    """censor_composed_dialogue preserves dialogue_options=None in QuestChoiceSchema."""
    censor = Settings().censor_word
    dialogue = ComposedDialogue(
        intent=Quest(type="Quest", objective="Deliver", has_choice=True),
        dialogue="A badword quest.",
        player_options=QuestChoiceSchema(
            accept="Accept badword",
            refuse="Refuse curse",
            dialogue_options=None,
        ),
    )

    censored = guardrail.censor_composed_dialogue(dialogue)

    assert censored.player_options.dialogue_options is None
    assert censored.player_options.accept == f"Accept {censor}"
    assert censored.player_options.refuse == f"Refuse {censor}"


def test_censor_composed_dialogue_with_dialogue_options_schema(guardrail):
    """censor_composed_dialogue redacts options in DialogueOptionsSchema."""
    censor = Settings().censor_word
    dialogue = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=True),
        dialogue="Beware the curse.",
        player_options=DialogueOptionsSchema(
            dialogue_options=["What curse?", "I fear no badword!"]
        ),
    )

    censored = guardrail.censor_composed_dialogue(dialogue)

    assert censored.dialogue == f"Beware the {censor}."
    assert isinstance(censored.player_options, DialogueOptionsSchema)
    assert censored.player_options.dialogue_options == [
        f"What {censor}?",
        f"I fear no {censor}!",
    ]


def test_censor_composed_dialogue_with_no_options(guardrail):
    """censor_composed_dialogue functions properly when player_options is None."""
    censor = Settings().censor_word
    dialogue = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="A solitary badword was spoken.",
        player_options=None,
    )

    censored = guardrail.censor_composed_dialogue(dialogue)

    assert censored.dialogue == f"A solitary {censor} was spoken."
    assert censored.player_options is None


def test_censor_composed_dialogue_clean_unmodified(guardrail):
    """censor_composed_dialogue leaves completely clean dialogue and options unchanged."""
    dialogue = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="Greetings honorable traveler.",
        player_options=DialogueOptionsSchema(dialogue_options=["Greetings friend."]),
    )

    censored = guardrail.censor_composed_dialogue(dialogue)

    assert censored.dialogue == dialogue.dialogue
    assert censored.player_options == dialogue.player_options

def test_composed_output_to_text_includes_player_options():
    d = ComposedDialogue(
        intent=Quest(type="Quest", objective="x", has_choice=True),
        dialogue="Main text",
        player_options=QuestChoiceSchema(accept="Yes", refuse="No", dialogue_options=["Opt A", "Opt B"]),
    )
    text = Guardrail._composed_output_to_text(d)
    for chunk in ("Main text", "Yes", "No", "Opt A", "Opt B"):
        assert chunk in text

def test_validate_composed_output_dirty_only_in_options(guardrail):
    d = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=True),
        dialogue="Clean text.",
        player_options=DialogueOptionsSchema(dialogue_options=["You badword!"]),
    )
    assert guardrail.validate_composed_output(d) is False

# --- Streaming Scanner & Concurrency Tests ---

def test_get_streaming_scanner(guardrail):
    """get_streaming_scanner returns an active StreamingLexiconScanner tied to current lexicon."""
    scanner = guardrail.get_streaming_scanner()
    assert isinstance(scanner, StreamingLexiconScanner)

    # Scan clean chunk
    assert scanner.feed("Hello brave ") == []
    # Feed dirty word
    matches = scanner.feed("badword!")
    assert matches == ["badword"]


def test_scanner_thread_safety(guardrail):
    """Retrieving streaming scanner and rebuilding lexicon is thread-safe under scanner_lock."""
    errors = []

    def reader():
        try:
            for _ in range(50):
                scanner = guardrail.get_streaming_scanner()
                assert scanner is not None
        except Exception as e:
            errors.append(e)

    def writer():
        try:
            for _ in range(20):
                guardrail._build_lexicon(Language.ENGLISH)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety errors occurred: {errors}"


# --- Helpers & Composed Output To Text Tests ---

def test_composed_output_to_text_with_list_and_str_options():
    """_composed_output_to_text handles string and list fields cleanly."""
    dialogue_with_str = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="Hello world",
        player_options=None,
    )
    text = Guardrail._composed_output_to_text(dialogue_with_str)
    assert text == "Hello world"

def test_debug(guardrail):
    d = ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=True),
        dialogue="Beware the curse.",
        player_options=DialogueOptionsSchema(dialogue_options=["What curse?", "I fear no badword!"]),
    )
    print(repr(Guardrail._composed_output_to_text(d)))
    print(guardrail.retrieve_banned_words_in_composed_dialogue(d))