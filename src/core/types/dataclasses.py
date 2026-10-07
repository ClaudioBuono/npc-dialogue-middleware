from dataclasses import dataclass, field
from typing import Any, Iterator
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
class DialogueStream:
    """A started dialogue stream together with its response headers.

    Attributes:
        chunks: Iterator yielding the dialogue text chunks. It is single-use:
            once consumed it cannot be replayed. If an unhandled error occurs
            during generation, it yields a final "[STREAM_ERROR]" marker
            instead of raising.
        headers: HTTP headers to attach to the response (e.g. profanity
            warnings). Empty if there is nothing to report.
    """

    chunks: Iterator[str]


# --- JUDGER DATACLASSES ---

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

class JudgeProblem(BaseModel):
    """A problem found by the Judger."""
    id: str
    answer: bool
    reason: str


class JudgeOutput(BaseModel):
    """Top-level structure returned by the judge LLM."""
    answers: list[JudgeProblem]