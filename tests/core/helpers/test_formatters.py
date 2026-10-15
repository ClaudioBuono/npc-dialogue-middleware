"""Unit tests for core.helpers.formatters – prompt text formatting, serialization, and context rendering."""
import json
from dataclasses import dataclass
import pytest
from pydantic import BaseModel

from api.schemas import ComposedDialogue, DialogueOptionsSchema, QuestChoiceSchema
from core.helpers.formatters import (
    _format_intent,
    _format_player_options,
    format_composed_dialogue,
    format_dialogue_history,
    format_game_context,
    format_intent_data,
    format_judge_issues,
    format_judge_questions,
    format_npc_context,
    format_stream,
    to_json_format,
)
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.dataclasses import JudgeIssue, JudgeQuestion


# ---------------------------------------------------------------------------
# TestFormatDialogueHistory
# ---------------------------------------------------------------------------
class TestFormatDialogueHistory:
    """Tests for format_dialogue_history rendering conversation logs."""

    def test_empty_history_returns_empty_string(self):
        """Empty history list produces empty string."""
        assert format_dialogue_history([], "Garrick") == ""

    def test_empty_turns_are_skipped(self):
        """Empty dictionary turns in history list are skipped."""
        history = [{}, {"NPC": "Hello!"}, {}]
        result = format_dialogue_history(history, "Garrick")
        assert result == '- Garrick: "Hello!"'

    def test_npc_turn_formatted_with_quotes(self):
        """NPC turn maps 'NPC' speaker to sanitized npc_name with quoted content."""
        history = [{"NPC": "Stay awhile and listen."}]
        result = format_dialogue_history(history, "Deckard Cain")
        assert result == '- Deckard Cain: "Stay awhile and listen."'

    def test_player_turn_formatted_without_quotes(self):
        """Player turn is labeled as 'Player' without surrounding double quotes."""
        history = [{"Player": "Tell me about the cathedral."}]
        result = format_dialogue_history(history, "Deckard Cain")
        assert result == "- Player: Tell me about the cathedral."

    def test_non_npc_speaker_defaults_to_player(self):
        """Any speaker key other than 'NPC' defaults to the Player format."""
        history = [{"Adventurer": "I seek glory."}]
        result = format_dialogue_history(history, "Garrick")
        assert result == "- Player: I seek glory."

    def test_alternating_dialogue_history(self):
        """A sequence of alternating turns is joined with newline characters."""
        history = [
            {"NPC": "Greetings traveler."},
            {"Player": "Do you have any work?"},
            {"NPC": "The rats in the cellar must be cleared."},
        ]
        result = format_dialogue_history(history, "Innkeeper")
        expected = (
            '- Innkeeper: "Greetings traveler."\n'
            "- Player: Do you have any work?\n"
            '- Innkeeper: "The rats in the cellar must be cleared."'
        )
        assert result == expected

    def test_npc_name_sanitized_and_truncated(self):
        """NPC name with tags, control chars, and >80 characters is properly sanitized."""
        dirty_long_name = "<Lord>\n" + "A" * 100
        history = [{"NPC": "Greetings."}]
        result = format_dialogue_history(history, dirty_long_name)
        assert "<" not in result
        assert "\n" not in result.split(":")[0]  # Name part contains no newlines
        # The safe name is truncated to 80 chars
        assert "‹Lord› " in result

    def test_content_sanitized_newlines_collapsed(self):
        """Turn content with line breaks and tags is flattened and angle brackets replaced."""
        history = [
            {"NPC": "First line\nSecond line <tag>"},
            {"Player": "Choice 1\r\nChoice 2"},
        ]
        result = format_dialogue_history(history, "Guide")
        assert '\nSecond line' not in result
        assert "‹tag›" in result
        assert "First line Second line ‹tag›" in result
        assert "Choice 1 Choice 2" in result


# ---------------------------------------------------------------------------
# TestToJsonFormat
# ---------------------------------------------------------------------------
class TestToJsonFormat:
    """Tests for to_json_format pretty-printer."""

    def test_dict_and_list_serialization(self):
        """Python dict and list are pretty-printed with 2-space indentation."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        result = to_json_format(data)
        assert json.loads(result) == data
        assert "  \"key\": \"value\"" in result

    def test_dataclass_serialization(self):
        """Dataclasses are converted via dataclasses.asdict."""
        question = JudgeQuestion(id="grammar", text="Is grammar correct?")
        result = to_json_format(question)
        parsed = json.loads(result)
        assert parsed == {"id": "grammar", "text": "Is grammar correct?"}

    def test_pydantic_model_serialization(self):
        """Pydantic BaseModel instances are converted via model_dump()."""
        dialogue = Dialogue(must_use_expression="Behold", number_of_options=1)
        result = to_json_format(dialogue)
        parsed = json.loads(result)
        assert parsed["type"] == "Dialogue"
        assert parsed["must_use_expression"] == "Behold"
        assert parsed["number_of_options"] == 1

    def test_nested_complex_structures(self):
        """Nested structures containing combinations of dicts, dataclasses, and models serialize."""
        @dataclass
        class Container:
            title: str
            issue: JudgeIssue

        container = Container(
            title="Report",
            issue=JudgeIssue(category="style", issue="Too verbose"),
        )
        result = to_json_format(container)
        parsed = json.loads(result)
        assert parsed["title"] == "Report"
        assert parsed["issue"] == {"category": "style", "issue": "Too verbose"}

    def test_unserializable_object_falls_back_to_str(self):
        """Objects that are neither dataclasses nor models fall back to str(o)."""
        class CustomNonSerializable:
            def __str__(self):
                return "CustomObjectString"

        data = {"custom": CustomNonSerializable()}
        result = to_json_format(data)
        parsed = json.loads(result)
        assert parsed["custom"] == "CustomObjectString"

    def test_unicode_characters_not_escaped(self):
        """Non-ASCII unicode characters are preserved without \\u escaping (ensure_ascii=False)."""
        data = {"greeting": "Buongiorno, cómo estás, grüß Gott"}
        result = to_json_format(data)
        assert "Buongiorno, cómo estás, grüß Gott" in result
        assert "\\u" not in result


# ---------------------------------------------------------------------------
# TestFormatStream
# ---------------------------------------------------------------------------
class TestFormatStream:
    """Tests for format_stream JSON formatter with plain text fallback."""

    def test_valid_json_formatted_indented(self):
        """Valid JSON string is formatted as 2-space indented JSON."""
        raw = '{"name":"Garrick","role":"blacksmith"}'
        result = format_stream(raw)
        expected = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        assert result == expected

    def test_valid_json_array_formatted(self):
        """Valid JSON array is formatted."""
        raw = '["item1","item2"]'
        result = format_stream(raw)
        assert result == '[\n  "item1",\n  "item2"\n]'

    def test_invalid_json_text_returned_unchanged(self):
        """Plain dialogue text or malformed JSON is returned as-is."""
        plain_text = "I have a bad feeling about this, traveler..."
        assert format_stream(plain_text) == plain_text

    def test_malformed_json_returned_unchanged(self):
        """Incomplete or truncated JSON stream returns original text."""
        malformed = '{"incomplete": '
        assert format_stream(malformed) == malformed

    def test_empty_string_returned_as_is(self):
        """Empty string returns empty string."""
        assert format_stream("") == ""

    def test_unicode_in_json_stream_preserved(self):
        """Accented/unicode characters in valid JSON stream are preserved."""
        raw = '{"city":"Zürich"}'
        result = format_stream(raw)
        assert "Zürich" in result


# ---------------------------------------------------------------------------
# TestFormatJudgeQuestionsAndIssues
# ---------------------------------------------------------------------------
class TestFormatJudgeQuestionsAndIssues:
    """Tests for format_judge_questions and format_judge_issues."""

    def test_format_judge_questions_empty(self):
        """Empty list of questions returns empty string."""
        assert format_judge_questions([]) == ""

    def test_format_judge_questions_single_and_multiple(self):
        """Questions are numbered 1-indexed with capitalized IDs."""
        questions = [
            JudgeQuestion(id="language", text="Is the dialogue in the correct language?"),
            JudgeQuestion(id="faithfulness", text="Does it respect the NPC lore?"),
        ]
        result = format_judge_questions(questions)
        expected = (
            "1. [Language] Is the dialogue in the correct language?\n"
            "2. [Faithfulness] Does it respect the NPC lore?"
        )
        assert result == expected

    def test_format_judge_issues_empty(self):
        """Empty list of issues returns empty string."""
        assert format_judge_issues([]) == ""

    def test_format_judge_issues_single_and_multiple(self):
        """Issues are formatted with 1-indexed numbering and capitalized category tag."""
        issues = [
            JudgeIssue(category="language", issue="Spoke English instead of Italian."),
            JudgeIssue(category="tone", issue="Too aggressive for a friendly shopkeeper."),
        ]
        result = format_judge_issues(issues)
        expected = (
            "1. [Language] Spoke English instead of Italian.\n"
            "2. [Tone] Too aggressive for a friendly shopkeeper."
        )
        assert result == expected


# ---------------------------------------------------------------------------
# TestFormatIntent
# ---------------------------------------------------------------------------
class TestFormatIntent:
    """Tests for internal _format_intent helper rendering Dialogue and Quest intents."""

    def test_dialogue_intent_minimal(self):
        """Minimal Dialogue intent formatting."""
        intent = Dialogue(number_of_options=0)
        result = _format_intent(intent)
        lines = result.split("\n")
        assert lines[0] == "Intent: DIALOGUE"
        assert lines[1] == "- Should offer extra dialogue options: no"

    def test_dialogue_intent_with_all_fields(self):
        """Dialogue intent with must_use_expression, more_info, and positive number_of_options."""
        intent = Dialogue(
            must_use_expression="By the gods!",
            more_info="The NPC knows the secret entrance.",
            number_of_options=2,
        )
        result = _format_intent(intent)
        assert 'Intent: DIALOGUE' in result
        assert '- MUST USE the following expression: "By the gods!"' in result
        assert "- Additional context: The NPC knows the secret entrance." in result
        assert "- Should offer extra dialogue options: yes" in result

    def test_quest_intent_all_fields(self):
        """Quest intent includes objective, optional narrative fields, and choice availability."""
        intent = Quest(
            objective="Retrieve the stolen ruby",
            name="The Crimson Gem",
            description="A thief snatched the gem from the vault.",
            location="Bandit Camp",
            reward="500 Gold Coins",
            has_choice=True,
            must_use_expression="Don't let them escape with it!",
            more_info="Guards are patrolling nearby.",
            number_of_options=1,
        )
        result = _format_intent(intent)
        assert "Intent: QUEST" in result
        assert '- MUST USE the following expression: "Don\'t let them escape with it!"' in result
        assert "- Additional context: Guards are patrolling nearby." in result
        assert "- Should offer extra dialogue options: yes" in result
        assert "- Objective: Retrieve the stolen ruby" in result
        assert "- Quest name: The Crimson Gem" in result
        assert "- Description: A thief snatched the gem from the vault." in result
        assert "- Location: Bandit Camp" in result
        assert "- Reward: 500 Gold Coins" in result
        assert "- Player can accept/decline: yes" in result

    def test_quest_intent_omits_empty_optional_fields(self):
        """Quest intent with missing optional fields omits them and reflects has_choice=False."""
        intent = Quest(
            objective="Escort the merchant",
            has_choice=False,
            number_of_options=0,
        )
        result = _format_intent(intent)
        assert "- Objective: Escort the merchant" in result
        assert "- Quest name:" not in result
        assert "- Description:" not in result
        assert "- Location:" not in result
        assert "- Reward:" not in result
        assert "- Player can accept/decline: no" in result


# ---------------------------------------------------------------------------
# TestFormatPlayerOptions
# ---------------------------------------------------------------------------
class TestFormatPlayerOptions:
    """Tests for internal _format_player_options helper."""

    def test_none_returns_empty_string(self):
        """None player options returns empty string."""
        assert _format_player_options(None) == ""

    def test_quest_choice_schema_without_dialogue_options(self):
        """QuestChoiceSchema formats accept and refuse choices."""
        options = QuestChoiceSchema(
            accept="I will aid you.",
            refuse="Find someone else.",
            dialogue_options=None,
        )
        result = _format_player_options(options)
        expected = (
            "Player options (quest choice):\n"
            '- Accept: "I will aid you."\n'
            '- Refuse: "Find someone else."'
        )
        assert result == expected

    def test_quest_choice_schema_with_dialogue_options(self):
        """QuestChoiceSchema formats accept, refuse, and additional dialogue options."""
        options = QuestChoiceSchema(
            accept="I accept.",
            refuse="I decline.",
            dialogue_options=["Where are they hiding?", "What's the pay?"],
        )
        result = _format_player_options(options)
        assert "Player options (quest choice):" in result
        assert '- Accept: "I accept."' in result
        assert '- Refuse: "I decline."' in result
        assert "- Additional options:" in result
        assert '  - "Where are they hiding?"' in result
        assert '  - "What\'s the pay?"' in result

    def test_dialogue_options_schema(self):
        """DialogueOptionsSchema formats a list of dialogue choices."""
        options = DialogueOptionsSchema(
            dialogue_options=["Tell me about your past.", "Farewell."],
        )
        result = _format_player_options(options)
        expected = (
            "Player options\n"
            '- "Tell me about your past."\n'
            '- "Farewell."'
        )
        assert result == expected

    def test_unsupported_options_type_returns_empty_string(self):
        """Object lacking expected attributes returns empty string."""
        class Unsupported:
            pass

        assert _format_player_options(Unsupported()) == ""


# ---------------------------------------------------------------------------
# TestFormatComposedDialogue
# ---------------------------------------------------------------------------
class TestFormatComposedDialogue:
    """Tests for format_composed_dialogue function."""

    def test_format_composed_dialogue_with_intent_and_options(self):
        """Formats intent, dialogue line, and player options separated by double newlines."""
        composed = ComposedDialogue(
            intent=Dialogue(must_use_expression="Watch out!", number_of_options=1),
            dialogue="There are monsters in the woods.",
            player_options=DialogueOptionsSchema(dialogue_options=["I am ready.", "I will flee."]),
        )
        result = format_composed_dialogue(composed, include_intent=True)
        sections = result.split("\n\n")
        assert len(sections) == 3
        assert "Intent: DIALOGUE" in sections[0]
        assert sections[1] == "Dialogue line:\nThere are monsters in the woods."
        assert "Player options" in sections[2]

    def test_format_composed_dialogue_without_intent(self):
        """When include_intent is False, intent section is omitted."""
        composed = ComposedDialogue(
            intent=Dialogue(),
            dialogue="Welcome to my shop.",
            player_options=None,
        )
        result = format_composed_dialogue(composed, include_intent=False)
        assert "Intent:" not in result
        assert result == "Dialogue line:\nWelcome to my shop."

    def test_format_composed_dialogue_without_player_options(self):
        """When player_options is None, player options block is omitted."""
        composed = ComposedDialogue(
            intent=Dialogue(),
            dialogue="I have nothing more to say.",
            player_options=None,
        )
        result = format_composed_dialogue(composed, include_intent=True)
        assert "Player options" not in result
        assert "Dialogue line:\nI have nothing more to say." in result


# ---------------------------------------------------------------------------
# TestFormatNPCContext
# ---------------------------------------------------------------------------
class TestFormatNPCContext:
    """Tests for format_npc_context rendering NPC profiles."""

    def test_minimal_npc_context(self):
        """Minimal NPC profile with required fields only."""
        npc = NPCContext(
            intent=Dialogue(),
            name="Boran",
            age=32,
            personality="Stoic and observant",
            context="Standing watch at the city gate",
            talkativeness=Talkativeness.AVERAGE,
            main_character_relation="Neutral guard duty",
        )
        result = format_npc_context(npc)
        lines = result.split("\n")
        assert "- Name: Boran" in lines
        assert "- Age: 32" in lines
        assert "- Personality: Stoic and observant" in lines
        assert "- Current context: Standing watch at the city gate" in lines
        assert "- Talkativeness: Balanced" in lines
        assert "- Relation to main character: Neutral guard duty" in lines
        assert not any(line.startswith("- Visual description:") for line in lines)
        assert not any(line.startswith("- Backstory:") for line in lines)
        assert not any(line.startswith("- Recent plot:") for line in lines)
        assert not any(line.startswith("- Spoken languages:") for line in lines)

    def test_full_npc_context(self):
        """Full NPC profile with all optional narrative attributes."""
        npc = NPCContext(
            intent=Dialogue(),
            name="Lady Elaria",
            age=45,
            personality="Cunning and calculating",
            context="Sitting in the high council chamber",
            talkativeness=Talkativeness.HIGH,
            main_character_relation="Distrustful ally",
            visual_description="Dressed in ornate velvet robes, wearing a gold signet ring",
            backstory="Inherited the dukedom after the war and consolidated power",
            recent_plot="Assassination attempt survived two nights ago",
            language=["Common", "High Elven"],
        )
        result = format_npc_context(npc)
        assert "- Name: Lady Elaria" in result
        assert "- Talkativeness: Talkative" in result
        assert "- Visual description: Dressed in ornate velvet robes, wearing a gold signet ring" in result
        assert "- Backstory: Inherited the dukedom after the war and consolidated power" in result
        assert "- Recent plot relevant to this NPC: Assassination attempt survived two nights ago" in result
        assert "- Spoken languages: Common, High Elven" in result

    def test_npc_context_field_sanitization(self):
        """Untrusted text in NPCContext fields is sanitized against tag and newline injection."""
        npc = NPCContext(
            intent=Dialogue(),
            name="<Evil\nKing>",
            age=100,
            personality="Mad\n\rscientist <danger>",
            context="In <the> dungeon",
            talkativeness=Talkativeness.VERY_LOW,
            main_character_relation="Nemesis\n<kill>",
        )
        result = format_npc_context(npc)
        assert "<" not in result
        assert ">" not in result
        assert "‹Evil King›" in result
        assert "Mad scientist ‹danger›" in result
        assert "‹the›" in result
        assert "Nemesis ‹kill›" in result


# ---------------------------------------------------------------------------
# TestFormatGameContext
# ---------------------------------------------------------------------------
class TestFormatGameContext:
    """Tests for format_game_context world state rendering."""

    def test_game_context_without_main_character(self):
        """Game context renders epoch, environment, and world state."""
        gc = GameContext(
            epoch="Late Bronze Age",
            environment="Arid desert dotted with ancient stone ruins",
            world_state="Drought has dried up major rivers and kingdoms are on the brink of war",
        )
        result = format_game_context(gc)
        lines = result.split("\n")
        assert "- Epoch: Late Bronze Age" in lines
        assert "- Environment: Arid desert dotted with ancient stone ruins" in lines
        assert "- Current world state: Drought has dried up major rivers and kingdoms are on the brink of war" in lines
        assert not any("Main character appearance:" in line for line in lines)

    def test_game_context_with_main_character_description(self):
        """Game context includes main character appearance when present."""
        gc = GameContext(
            epoch="Cyberpunk 2099",
            environment="Rainy megacity with neon-drenched skyscrapers",
            world_state="Corporate war has broken out in the lower sectors",
            main_character_description="Cyborg detective wearing a tattered trench coat",
        )
        result = format_game_context(gc)
        assert "- Main character appearance: Cyborg detective wearing a tattered trench coat" in result

    def test_game_context_field_sanitization(self):
        """Tags and control chars in GameContext fields are sanitized."""
        gc = GameContext(
            epoch="<Sci-Fi>\nEra",
            environment="Dark <neon> streets\nwith acid rain",
            world_state="AI <takeover>\ncompleted",
            main_character_description="Rebel <leader>\narmed with EMP",
        )
        result = format_game_context(gc)
        assert "<" not in result
        assert ">" not in result
        assert "‹Sci-Fi› Era" in result
        assert "Dark ‹neon› streets with acid rain" in result
        assert "AI ‹takeover› completed" in result
        assert "Rebel ‹leader› armed with EMP" in result


# ---------------------------------------------------------------------------
# TestFormatIntentData
# ---------------------------------------------------------------------------
class TestFormatIntentData:
    """Tests for format_intent_data data-only view."""

    def test_format_intent_data_dialogue(self):
        """Data-only view for Dialogue intent."""
        intent = Dialogue(
            must_use_expression="The stars are aligning",
            more_info="Keep voice hushed",
            number_of_options=3,
        )
        result = format_intent_data(intent)
        lines = result.split("\n")
        assert "- Type: Dialogue" in lines
        assert '- Required expression: "The stars are aligning"' in lines
        assert "- Additional context: Keep voice hushed" in lines
        # Does not include control flags
        assert not any("Should offer extra dialogue options" in line for line in lines)

    def test_format_intent_data_quest_full(self):
        """Data-only view for Quest intent includes all available quest metadata."""
        intent = Quest(
            objective="Retrieve the stolen chalice",
            name="The Sacred Grail",
            description="Taken by temple raiders",
            location="Catacombs",
            reward="Holy Relic",
            has_choice=True,
            must_use_expression="Hurry before dusk",
            more_info="Watch for traps",
        )
        result = format_intent_data(intent)
        assert "- Type: Quest" in result
        assert '- Required expression: "Hurry before dusk"' in result
        assert "- Additional context: Watch for traps" in result
        assert "- Objective: Retrieve the stolen chalice" in result
        assert "- Quest name: The Sacred Grail" in result
        assert "- Description: Taken by temple raiders" in result
        assert "- Location: Catacombs" in result
        assert "- Reward: Holy Relic" in result
        # Does not include control flags
        assert not any("Player can accept/decline" in line for line in result.split("\n"))

    def test_format_intent_data_quest_partial(self):
        """Data-only view for Quest intent renders only populated fields."""
        intent = Quest(
            objective="Investigate noises in basement",
            reward="5 Copper",
        )
        result = format_intent_data(intent)
        assert "- Type: Quest" in result
        assert "- Objective: Investigate noises in basement" in result
        assert "- Reward: 5 Copper" in result
        assert "- Quest name:" not in result
        assert "- Description:" not in result
        assert "- Location:" not in result
