"""Reply formats Apertus must follow, checked with pydantic.

Fields and choices follow Joswin's prompts (``src/reasoning``); the decision format is built
per scene so that object fields only accept ids from that scene's graph.
"""

from typing import Literal

from pydantic import BaseModel, create_model

from src.reasoning.decision import HOLD, PLACES
from src.reasoning.safety import MANNERS

NO_OBJECT_ID = "none"


class RegionName(BaseModel):
    """What one outlined region is."""

    name: str
    is_object: bool


class SafetyScreen(BaseModel):
    """Facts about the instruction that decide whether planning may start."""

    analysis: str
    manner: Literal[tuple(MANNERS)]
    involves_person: bool
    contradicts_itself: bool
    message: str


def decision_model(object_ids: list[str]) -> type[BaseModel]:
    """Build the decision format for one scene, with object fields limited to its ids.

    Args:
        object_ids: ids from the scene graph.

    Returns:
        Pydantic class with ``analysis``, ``decision``, ``message`` and ``moves``; each move has
        an ``object`` id and a ``destination`` (a place, ``hold`` or an object id).
    """
    targets = tuple(sorted(set(object_ids))) or (NO_OBJECT_ID,)
    move = create_model(
        "Move",
        object=(Literal[targets], ...),
        destination=(Literal[(*PLACES, HOLD, *targets)], ...),
    )
    return create_model(
        "Decision",
        analysis=(str, ...),
        decision=(Literal["plan", "ask_user", "refuse"], ...),
        message=(str, ...),
        moves=(list[move], ...),
    )
