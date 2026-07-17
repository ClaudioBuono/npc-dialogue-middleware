from dataclasses import dataclass, field
from typing import Any, List

from core.types.contexts import Dialogue, Quest


@dataclass
class Contract:
    """
    Structured payload ready to be sent to the LLM.
    """
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any] = field(default_factory=dict)

@dataclass
class ComposedDialogue:
    """
    Structured dialogue to be sent as an output.
    """
    intent: Dialogue | Quest
    dialogue: str
    options: List[str] | None
    accept: str | None
    refuse: str | None

