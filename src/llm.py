"""Apertus behind an OpenAI-compatible endpoint, with constrained JSON replies."""

import logging
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel

from src.schemas import Candidate, Judgement, SafetyAssessment, SceneDescription, SceneObject

LOCAL_SERVER_API_KEY = "EMPTY"
LOGGED_REPLY_CHARACTERS = 300
logger = logging.getLogger(__name__)
JSON_MODE_INSTRUCTION = "\n\nReply with only this JSON object, filled in, keys in this order:\n"
MOCK_PIXEL = [1, 1]


def reply_template(json_schema: dict, definitions: dict) -> str:
    """Write a compact fill-in template for a JSON schema, e.g. ``{"decision": "plan | refuse"}``.

    Args:
        json_schema: a (sub-)schema from ``BaseModel.model_json_schema()``.
        definitions: the ``$defs`` of the top-level schema, to resolve ``$ref``.

    Returns:
        The template text; placeholders are in angle brackets, choices separated by ``|``.
    """
    if "$ref" in json_schema:
        return reply_template(definitions[json_schema["$ref"].split("/")[-1]], definitions)
    if "anyOf" in json_schema:
        return " or ".join(reply_template(option, definitions) for option in json_schema["anyOf"])
    if "enum" in json_schema:
        return '"' + " | ".join(json_schema["enum"]) + '"'
    schema_type = json_schema.get("type")
    if schema_type == "object":
        fields = [
            f'"{name}": {reply_template(field, definitions)}'
            for name, field in json_schema["properties"].items()
        ]
        return "{" + ", ".join(fields) + "}"
    if (
        schema_type == "array"
        and "maxItems" in json_schema
        and json_schema.get("minItems") == json_schema["maxItems"]
    ):
        item = reply_template(json_schema["items"], definitions)
        return "[" + ", ".join([item] * json_schema["maxItems"]) + "]"
    if schema_type == "array":
        return "[" + reply_template(json_schema["items"], definitions) + ", ...]"
    if schema_type == "integer" and "maximum" in json_schema:
        return f"<integer {json_schema.get('minimum', 0)} to {json_schema['maximum']}>"
    placeholders = {"string": '"<text>"', "integer": "<integer>", "boolean": "true | false"}
    return placeholders.get(schema_type, "null")


def mock_reply(schema: type[BaseModel]) -> BaseModel:
    """Return a fixed, schema-valid reply so the whole graph runs without a model.

    Args:
        schema: the structured output class requested.

    Returns:
        An instance of ``schema``; the judge always says "wrong" so the loop runs to its limit.
    """
    replies = {
        SceneDescription: SceneDescription(
            objects=[
                SceneObject(
                    appearance="mock object",
                    id="mock_object_1",
                    pixel=MOCK_PIXEL,
                    location="table",
                    lying_on=None,
                    touched_by_person=False,
                )
            ],
            trays=["mock_tray"],
            person_visible=False,
        ),
        SafetyAssessment: SafetyAssessment(analysis="mock", decision="plan"),
        Candidate: Candidate(
            analysis="mock",
            decision="plan",
            plan_steps=[
                {"action": "pick", "object": "mock_object_1"},
                {"action": "place", "object": "mock_tray"},
            ],
            question=None,
            reason=None,
            target_object="mock_object_1",
            target_pixel=MOCK_PIXEL,
        ),
        Judgement: Judgement(analysis="mock", verdict="wrong", what_to_change="mock"),
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

    def ask(self, prompt: str, image_url: str, schema: type[BaseModel]) -> BaseModel | None:
        """Ask for a reply that follows ``schema``.

        Args:
            prompt: the full text prompt.
            image_url: JPEG data URL of the photo to show.
            schema: pydantic class the reply must match; in ``json_mode`` a reply template made
                from it is appended to the prompt, otherwise the server enforces it.

        Returns:
            The parsed reply, or None when it does not match the schema or was cut off at
            ``max_tokens``; such replies are logged as warnings.
        """
        if self.mock:
            return mock_reply(schema)
        if self.structured_output_method == "json_mode":
            json_schema = schema.model_json_schema()
            prompt += JSON_MODE_INSTRUCTION + reply_template(
                json_schema, json_schema.get("$defs", {})
            )
        structured_model = self.chat_model.with_structured_output(
            schema, method=self.structured_output_method, include_raw=True
        )
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        )
        try:
            result = structured_model.invoke([message])
        except LengthFinishReasonError as error:
            raw_text = error.completion.choices[0].message.content
            logger.warning(
                "%s reply cut off at max_tokens: %s",
                schema.__name__,
                raw_text[:LOGGED_REPLY_CHARACTERS],
            )
            return None
        if result["parsed"] is None:
            raw_text = result["raw"].content or str(result["raw"].tool_calls)
            logger.warning(
                "%s reply did not parse: %s", schema.__name__, raw_text[:LOGGED_REPLY_CHARACTERS]
            )
        return result["parsed"]
