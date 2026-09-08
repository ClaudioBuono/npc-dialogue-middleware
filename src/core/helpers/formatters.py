import dataclasses
import json

from pydantic import BaseModel

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