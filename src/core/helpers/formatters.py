import dataclasses
import json
from typing import List, Union

from pydantic import BaseModel

from api.schemas import ComposedDialogue
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest
from core.types.dataclasses import JudgeQuestion


def format_dialogue_history(history: list[dict[str, str]], npc_name: str) -> str:
    """Format the raw dialogue history into a human-readable block for the prompt.

    Args:
        history: List of single-key dicts, e.g. [{'NPC': '...'}, {'Player': '...'}].
            The key indicates the speaker ('NPC' or 'Player'), the value is
            the line spoken or the label of the choice made.
        npc_name: Display name of the NPC, used in place of the generic 'NPC' key.

    Returns:
        A formatted multi-line string ready to be embedded in the prompt.
        Returns an empty string if history is empty.
    """
    if not history:
        return ""

    lines = []
    for turn in history:
        if not turn:
            continue
        speaker, content = next(iter(turn.items()))
        content = content.strip()

        if speaker == "NPC":
            lines.append(f'- {npc_name}: "{content}"')
        else:
            lines.append(f"- Player: {content}")

    return "\n".join(lines)


def to_json_format(obj) -> str:
    """Pretty-print an object (dict, dataclass, or Pydantic model) as a JSON string.

    Args:
        obj: The object to serialize. Can be a dict, list, dataclass,
            Pydantic model, or any combination thereof.

    Returns:
        str: An indented (2-space) JSON string representation of ``obj``.
    """
    def custom_encoder(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, BaseModel):
            return o.model_dump()
        return str(o)
    return json.dumps(obj, default=custom_encoder, indent=2, ensure_ascii=False)


def format_composed_dialogue(composed_dialogue: ComposedDialogue) -> str:
    """Format a ComposedDialogue instance into human-readable text for an LLM prompt.

    Args:
        composed_dialogue (ComposedDialogue): The composed dialogue object to format.

    Returns:
        str: Structured textual representation formatted with markdown sections.
    """
    sections = [
        _format_intent(composed_dialogue.intent),
        f"Dialogue line:\n{composed_dialogue.dialogue}",
    ]

    options_block = _format_player_options(composed_dialogue.player_options)
    if options_block:
        sections.append(options_block)

    return "\n\n".join(sections)

def format_judge_questions(questions: List[JudgeQuestion]) -> str:
    """
    Format a list of questions into human-readable text for an LLM prompt.
    """
    formatted_questions: List[str] = []

    i = 0
    for question in questions:
        i += 1
        formatted = f"{i}. [{question.id.capitalize()}] {question.text}"
        formatted_questions.append(formatted)

    return "\n".join(formatted_questions)

def _format_intent(intent: Union[Quest, Dialogue]) -> str:
    """Format a conversation intent (Quest or Dialogue) into key-value prompt lines.

    Args:
        intent (Union[Quest, Dialogue]): The dialogue intent or quest payload.

    Returns:
        str: Formatted string listing intent metadata, restrictions, and quest fields.
    """
    lines = [f"- Intent: {intent.type}"]

    if intent.must_use_expression:
        lines.append(f'- Must include this key message: "{intent.must_use_expression}"')
    if intent.more_info:
        lines.append(f"- Additional context: {intent.more_info}")
    lines.append(f"- Should offer extra dialogue options: {'yes' if intent.has_options else 'no'}")

    if intent.type == "Quest":
        lines.append(f"- Objective: {intent.objective}")
        if intent.name:
            lines.append(f"- Quest name: {intent.name}")
        if intent.description:
            lines.append(f"- Description: {intent.description}")
        if intent.location:
            lines.append(f"- Location: {intent.location}")
        if intent.reward:
            lines.append(f"- Reward: {intent.reward}")
        lines.append(f"- Player can accept/decline: {'yes' if intent.has_choice else 'no'}")

    return "\n".join(lines)


def _format_player_options(player_options) -> str:
    """Format player dialogue options or quest choices into prompt lines.

    Args:
        player_options: Schema object containing player choices/options, or None.

    Returns:
        str: Formatted options text block, or an empty string if no options exist.
    """
    if player_options is None:
        return ""

    if hasattr(player_options, "accept") and hasattr(player_options, "refuse"):
        lines = [
            "Player options (quest choice):",
            f'- Accept: "{player_options.accept}"',
            f'- Refuse: "{player_options.refuse}"',
        ]
        if player_options.dialogue_options:
            lines.append("- Additional options:")
            lines.extend(f'  - "{opt}"' for opt in player_options.dialogue_options)
        return "\n".join(lines)

    if hasattr(player_options, "dialogue_options"):
        lines = ["Player options"]
        lines.extend(f'- "{opt}"' for opt in player_options.dialogue_options)
        return "\n".join(lines)

    return ""


def format_npc_content(npc_context: NPCContext) -> str:
    """Format an NPCContext instance into human-readable text for an LLM prompt.

    Renders the NPC's core profile (name, age, personality, current state, relationship
    with the main character) along with optional context fields such as backstory,
    visual description, plot context, and allowed languages.

    Args:
        npc_context (NPCContext): The NPC context profile to format.

    Returns:
        str: Structured textual representation formatted with markdown list items.
    """
    lines = [
        f"- Name: {npc_context.name}",
        f"- Age: {npc_context.age}",
        f"- Personality: {npc_context.personality}",
        f"- Current context: {npc_context.context}",
        f"- Talkativeness: {npc_context.talkativeness.value}",
        f"- Relation to main character: {npc_context.main_character_relation}",
    ]

    if npc_context.visual_description:
        lines.append(f"- Visual description: {npc_context.visual_description}")
    if npc_context.backstory:
        lines.append(f"- Backstory: {npc_context.backstory}")
    if npc_context.recent_plot:
        lines.append(f"- Recent plot relevant to this NPC: {npc_context.recent_plot}")
    if npc_context.language:
        lines.append(f"- Spoken languages: {', '.join(npc_context.language)}")

    npc_block = "\n".join(lines)

    return f"{npc_block}"


def format_game_context(game_context: GameContext) -> str:
    """Format a GameContext instance into human-readable text for an LLM prompt.

    Includes global world information such as epoch, environment, current world state,
    and optional main character description.

    Args:
        game_context (GameContext): The global game context to format.

    Returns:
        str: Structured textual representation formatted with markdown list items.
    """
    lines = [
        f"- Epoch: {game_context.epoch}",
        f"- Environment: {game_context.environment}",
        f"- Current world state: {game_context.world_state}",
    ]

    if game_context.main_character_description:
        lines.append(f"- Main character appearance: {game_context.main_character_description}")

    return "\n".join(lines)