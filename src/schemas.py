"""Structured outputs Apertus must produce, used for constrained JSON decoding.

Field names sort alphabetically in the order the model should write them, because the hosted
endpoint emits keys alphabetically while a local vLLM server keeps the declared order.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Pixel = Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)]


class SceneObject(BaseModel):
    """One pickable object seen in the photo."""

    appearance: str
    id: str
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

    analysis: str
    decision: Literal["plan", "ask_user", "refuse"]


class PlanStep(BaseModel):
    """One pick or place action on an object id from the scene."""

    action: Literal["pick", "place"]
    object: str


class Candidate(BaseModel):
    """The actor's answer to one instruction, before conversion to the official schema.

    Fields that only one decision uses are optional, so the model can end its reply once the
    answer is complete instead of padding it.
    """

    analysis: str
    decision: Literal["plan", "ask_user", "refuse"]
    plan_steps: list[PlanStep] = Field(default_factory=list)
    question: str | None = None
    reason: str | None = None
    target_object: str | None = None
    target_pixel: Pixel | None = None


class Judgement(BaseModel):
    """The judge's verdict on one candidate."""

    analysis: str
    verdict: Literal["right", "wrong"]
    what_to_change: str
