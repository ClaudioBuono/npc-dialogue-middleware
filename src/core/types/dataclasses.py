from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

@dataclass
class Contract:
    """
    Structured payload ready to be sent to the LLM.
    """
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class JudgeQuestion:
    """
    A question for the judge to evaluate.
    """
    id: str
    text: str

@dataclass(frozen=True)
class JudgeIssue:
    """An issue in the dialogue found by the Judge"""
    category: str
    issue: str


# TODO: move to a better place
class JudgeProblem(BaseModel):
    """A problem found by the Judger."""
    id: str
    answer: bool
    reason: str


class JudgeOutput(BaseModel):
    """Top-level structure returned by the judge LLM."""
    answers: list[JudgeProblem]