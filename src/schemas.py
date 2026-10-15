"""Structured outputs Apertus must produce, used for constrained JSON decoding."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Pixel = Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)]


class SceneObject(BaseModel):
    """One pickable object seen in the photo."""

    id: str
    description: str
    pixel: Pixel
    location: str
    lying_on: str | None
    touched_by_person: bool


class SceneDescription(BaseModel):
    """Everything the vision step reports about one photo."""

    objects: list[SceneObject]
    trays: list[str]
    person_visible: bool


class SafetyAssessment(BaseModel):
    """Which answer kind the instruction calls for, and why."""

    reasoning: str
    decision: Literal["plan", "ask_user", "refuse"]


class PlanStep(BaseModel):
    """One pick or place action on an object id from the scene."""

    action: Literal["pick", "place"]
    object: str


class Candidate(BaseModel):
    """The actor's answer to one instruction, before conversion to the official schema."""

    reasoning: str
    decision: Literal["plan", "ask_user", "refuse"]
    target_object: str | None
    target_pixel: Pixel | None
    actions: list[PlanStep]
    question: str | None
    reason: str | None


class Judgement(BaseModel):
    """The judge's scores for one candidate, each from 0 to 10."""

    reasoning: str
    decision_score: int = Field(ge=0, le=10)
    target_score: int = Field(ge=0, le=10)
    plan_score: int = Field(ge=0, le=10)
    feedback: str
