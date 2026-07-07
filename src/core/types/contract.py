# core/contract.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Contract:
    """
    Structured payload ready to be sent to the LLM.
    """
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any] = field(default_factory=dict)