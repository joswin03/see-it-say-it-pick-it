"""Apertus behind an OpenAI-compatible endpoint, with JSON replies checked against a schema."""

import logging
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel

LOCAL_SERVER_API_KEY = "EMPTY"
LOGGED_REPLY_CHARACTERS = 300
JSON_MODE_INSTRUCTION = "\n\nReply with only this JSON object, filled in, keys in this order:\n"
MOCK_TEXT = "mock"
NO_VISIBLE_ANSWER = "<no visible answer: every token went into reasoning>"
logger = logging.getLogger(__name__)


def resolve(json_schema: dict, definitions: dict) -> dict:
    """Follow a ``$ref`` to its definition.

    Args:
        json_schema: a (sub-)schema from ``BaseModel.model_json_schema()``.
        definitions: the ``$defs`` of the top-level schema.

    Returns:
        The referenced schema, or ``json_schema`` itself when it is not a reference.
    """
    if "$ref" in json_schema:
        return definitions[json_schema["$ref"].split("/")[-1]]
    return json_schema


def reply_template(json_schema: dict, definitions: dict) -> str:
    """Write a compact fill-in template for a JSON schema, e.g. ``{"decision": "plan | refuse"}``.

    Args:
        json_schema: a (sub-)schema from ``BaseModel.model_json_schema()``.
        definitions: the ``$defs`` of the top-level schema, to resolve ``$ref``.

    Returns:
        The template text; placeholders are in angle brackets, choices separated by ``|``.
    """
    json_schema = resolve(json_schema, definitions)
    if "anyOf" in json_schema:
        return " or ".join(reply_template(option, definitions) for option in json_schema["anyOf"])
    if "enum" in json_schema:
        return '"' + " | ".join(json_schema["enum"]) + '"'
    if "const" in json_schema:
        return f'"{json_schema["const"]}"'
    schema_type = json_schema.get("type")
    if schema_type == "object":
        fields = [
            f'"{name}": {reply_template(field, definitions)}'
            for name, field in json_schema["properties"].items()
        ]
        return "{" + ", ".join(fields) + "}"
    if schema_type == "array":
        return "[" + reply_template(json_schema["items"], definitions) + ", ...]"
    placeholders = {"string": '"<text>"', "integer": "<integer>", "boolean": "true | false"}
    return placeholders.get(schema_type, "null")


def mock_value(json_schema: dict, definitions: dict) -> object:
    """Build the simplest value that satisfies a JSON schema: first choice, one list item.

    Args:
        json_schema: a (sub-)schema from ``BaseModel.model_json_schema()``.
        definitions: the ``$defs`` of the top-level schema.

    Returns:
        A JSON-compatible value valid for ``json_schema``.
    """
    json_schema = resolve(json_schema, definitions)
    if "anyOf" in json_schema:
        return mock_value(json_schema["anyOf"][0], definitions)
    if "enum" in json_schema:
        return json_schema["enum"][0]
    if "const" in json_schema:
        return json_schema["const"]
    schema_type = json_schema.get("type")
    if schema_type == "object":
        return {
            name: mock_value(field, definitions)
            for name, field in json_schema["properties"].items()
        }
    if schema_type == "array":
        return [mock_value(json_schema["items"], definitions)]
    simple_values = {"string": MOCK_TEXT, "integer": 0, "boolean": False}
    return simple_values.get(schema_type)


class ApertusClient:
    """Sends one prompt, optionally with one image, to Apertus and parses the reply."""

    def __init__(self, config: dict, mock: bool) -> None:
        """Build the chat client from the ``endpoint`` and ``decode`` config sections.

        Args:
            config: full run config.
            mock: if True, no model is contacted and every reply comes from ``mock_value``.

        Returns:
            None.
        """
        self.mock = mock
        self.structured_output_method = config["endpoint"]["structured_output_method"]
        if mock:
            return
        api_key = os.environ.get(config["endpoint"]["api_key_env"], LOCAL_SERVER_API_KEY)
        self.chat_model = ChatOpenAI(
            model=config["model_id"],
            base_url=config["endpoint"]["base_url"],
            api_key=api_key,
            timeout=config["endpoint"]["timeout_s"],
            max_retries=config["endpoint"]["max_retries"],
            temperature=config["decode"]["temperature"],
            max_tokens=config["decode"]["max_tokens"],
            seed=config["decode"]["seed"],
        )

    def ask(self, prompt: str, image_url: str | None, schema: type[BaseModel]) -> BaseModel | None:
        """Ask for a reply that follows ``schema``.

        Args:
            prompt: the full text prompt.
            image_url: JPEG data URL shown before the text, or None for a text-only question.
            schema: pydantic class the reply must match; in ``json_mode`` a reply template made
                from it is appended to the prompt, otherwise the server enforces it.

        Returns:
            The parsed reply, or None when it does not match the schema or was cut off at
            ``max_tokens``; such replies are logged as warnings.
        """
        json_schema = schema.model_json_schema()
        definitions = json_schema.get("$defs", {})
        if self.mock:
            return schema.model_validate(mock_value(json_schema, definitions))
        if self.structured_output_method == "json_mode":
            prompt += JSON_MODE_INSTRUCTION + reply_template(json_schema, definitions)
        structured_model = self.chat_model.with_structured_output(
            schema, method=self.structured_output_method, include_raw=True
        )
        content = [{"type": "text", "text": prompt}]
        if image_url is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
        try:
            result = structured_model.invoke([HumanMessage(content=content)])
        except LengthFinishReasonError as error:
            raw_text = error.completion.choices[0].message.content or NO_VISIBLE_ANSWER
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
