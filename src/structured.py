"""Validate structured model replies and recover complete JSON missing closing brackets.

Adapted from Joswin's OpenCV branch; incomplete content is never fabricated.
"""

import json

from jsonschema import Draft202012Validator


class ModelReplyError(ValueError):
    """The model's reply was cut off by the token limit or is not valid JSON."""

    def __init__(self, message: str, raw: str) -> None:
        """Store the failed reply.

        Args:
            message: Failure description.
            raw: Original reply text.

        Returns:
            None.
        """
        super().__init__(message)
        self.raw = raw


CLOSING_SUFFIXES = ("}", "]}", "}]}")


def close_stalled_reply(raw: str, schema: dict) -> dict | None:
    """Recover a reply whose content is complete but whose closing brackets never came.

    Under constrained decoding Apertus sometimes finishes every field and then emits only
    whitespace until the token limit, instead of the final "}". If appending the missing
    closing brackets yields JSON with every required field, that JSON is the reply.

    Args:
        raw: reply text cut off at the token limit.
        schema: the schema the reply had to satisfy.

    Returns:
        The recovered reply, or None if the content itself is incomplete.
    """
    trimmed = raw.rstrip()
    for suffix in CLOSING_SUFFIXES:
        try:
            reply = json.loads(trimmed + suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(reply, dict) and not list(Draft202012Validator(schema).iter_errors(reply)):
            return reply
    return None
