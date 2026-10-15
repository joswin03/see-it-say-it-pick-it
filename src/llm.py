"""Apertus behind an OpenAI-compatible endpoint, with constrained JSON replies."""

import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.schemas import Candidate, Judgement, SafetyAssessment, SceneDescription, SceneObject

LOCAL_SERVER_API_KEY = "EMPTY"
MOCK_PIXEL = [1, 1]


def mock_reply(schema: type[BaseModel]) -> BaseModel:
    """Return a fixed, schema-valid reply so the whole graph runs without a model.

    Args:
        schema: the structured output class requested.

    Returns:
        An instance of ``schema``; the judge always scores 5 so the loop runs to its limit.
    """
    replies = {
        SceneDescription: SceneDescription(
            objects=[
                SceneObject(
                    id="mock_object_1",
                    description="mock object",
                    pixel=MOCK_PIXEL,
                    location="table",
                    lying_on=None,
                    touched_by_person=False,
                )
            ],
            trays=["mock_tray"],
            person_visible=False,
        ),
        SafetyAssessment: SafetyAssessment(reasoning="mock", decision="plan"),
        Candidate: Candidate(
            reasoning="mock",
            decision="plan",
            target_object="mock_object_1",
            target_pixel=MOCK_PIXEL,
            actions=[
                {"action": "pick", "object": "mock_object_1"},
                {"action": "place", "object": "mock_tray"},
            ],
            question=None,
            reason=None,
        ),
        Judgement: Judgement(
            reasoning="mock", decision_score=5, target_score=5, plan_score=5, feedback="mock"
        ),
    }
    return replies[schema]


class ApertusClient:
    """Sends one prompt plus one photo to Apertus and parses the constrained reply."""

    def __init__(self, config: dict, mock: bool) -> None:
        """Build the chat client from the ``endpoint`` and ``decode`` config sections.

        Args:
            config: full run config.
            mock: if True, no model is contacted and ``mock_reply`` answers.

        Returns:
            None.
        """
        self.mock = mock
        self.structured_output_method = config["endpoint"]["structured_output_method"]
        if mock:
            return
        api_key_env = config["endpoint"]["api_key_env"]
        api_key = os.environ[api_key_env] if api_key_env else LOCAL_SERVER_API_KEY
        self.chat_model = ChatOpenAI(
            model=config["model_id"],
            base_url=config["endpoint"]["base_url"],
            api_key=api_key,
            timeout=config["endpoint"]["timeout_s"],
            temperature=config["decode"]["temperature"],
            max_tokens=config["decode"]["max_tokens"],
            seed=config["decode"]["seed"],
        )

    def ask(
        self, prompt: str, image_url: str, schema: type[BaseModel]
    ) -> tuple[BaseModel | None, str]:
        """Ask for a reply that follows ``schema``.

        Args:
            prompt: the full text prompt.
            image_url: JPEG data URL of the photo to show.
            schema: pydantic class the reply must match.

        Returns:
            ``(parsed, raw_text)``; ``parsed`` is None when the reply does not match the schema.
        """
        if self.mock:
            reply = mock_reply(schema)
            return reply, reply.model_dump_json()
        structured_model = self.chat_model.with_structured_output(
            schema, method=self.structured_output_method, include_raw=True
        )
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        )
        result = structured_model.invoke([message])
        raw_message = result["raw"]
        raw_text = raw_message.content or str(raw_message.tool_calls)
        return result["parsed"], raw_text
