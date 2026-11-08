"""Pydantic models for LLM structured outputs (sent as response_schema) and pipeline results."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

QuestionType = Literal['direct', 'policy', 'procedural', 'decision', 'comparison']
QUESTION_TYPES = ('direct', 'policy', 'procedural', 'decision', 'comparison')
Operator = Literal['>=', '<=', '==', '!=', 'in', 'not_in']

# Gemini-facing schemas: no default values (the Gemini API rejects them); use Optional for "may be null".


class QuestionTypeOut(BaseModel):
    label: QuestionType


class UserFact(BaseModel):
    name: str
    value: Optional[str]
    type: Literal['number', 'bool', 'string']


class Decision(BaseModel):
    id: str
    kind: Literal['boolean', 'choice', 'score']
    question: str
    fact: Optional[str]
    operator: Optional[Operator]
    threshold: Optional[str]
    options: list[str]
    source_page: Optional[int]
    source_document: Optional[str]


class CombineSpec(BaseModel):
    rule: Literal['all_required', 'any', 'weighted']
    required_ids: list[str]


class Decomposition(BaseModel):
    user_facts: list[UserFact]
    decisions: list[Decision]
    combine: CombineSpec
    missing_facts: list[str]


class RatingOut(BaseModel):
    rating: int = Field(ge=0, le=4)


def choice_schema(options):
    """A one-field model whose `choice` is restricted (enum) to the given options."""
    return create_model('ChoiceOut', choice=(Literal[tuple(options)], ...))


# Internal results (never sent to Gemini).


class DecisionResult(BaseModel):
    id: str
    kind: str
    question: str = ''
    value: Any = None  # True/False/'unknown' (boolean), option (choice), 0–1 float (score)
    confidence: float = 0.0
    evidence_ref: Optional[str] = None
    error: Optional[str] = None  # safe, generic text; details go to server logs


class CombinedResult(BaseModel):
    verdict: Optional[Literal['eligible', 'not_eligible', 'needs_review']] = None
    outcome: Literal['answered', 'needs_review']
    summary: list[str] = []
    missing_facts: list[str] = []
