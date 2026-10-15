"""Minimal client for an OpenAI-compatible chat endpoint (vLLM serving Apertus).

Replies are constrained to a JSON schema through ``response_format``, so every answer parses.
"""

import base64
import json
import os
import urllib.request

import cv2
import numpy as np


class ModelReplyError(ValueError):
    """The model's reply was cut off by the token limit or is not valid JSON."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def image_url(rgb: np.ndarray, max_side: int) -> str:
    """Encode a photo as a JPEG data URL, downscaled so its long side is at most ``max_side``.

    Args:
        rgb: (H, W, 3) uint8 photo.
        max_side: longest side in pixels after scaling.

    Returns:
        ``data:image/jpeg;base64,...`` string.
    """
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        rgb = cv2.resize(
            rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


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
        if isinstance(reply, dict) and all(key in reply for key in schema["required"]):
            return reply
    return None


def api_key() -> str:
    """Key for a hosted endpoint from ``APERTUS_API_KEY`` (kept in the git-ignored ``.env``); local
    vLLM needs none."""
    return os.environ.get("APERTUS_API_KEY", "none")


def ask_json(
    endpoint: dict, prompt: str, images: list[str], schema: dict, max_tokens: int
) -> tuple[dict, str]:
    """Send one user turn with images and get a schema-constrained JSON reply.

    Args:
        endpoint: dict with ``base_url``, ``model_id`` and ``request_timeout_s``.
        prompt: user text.
        images: data URLs from ``image_url``, placed before the text.
        schema: JSON schema the reply must satisfy.
        max_tokens: generation limit.

    Returns:
        The parsed reply and the raw reply text (trailing whitespace of a stalled reply
        removed and marked with ``[closed]``, see ``close_stalled_reply``).

    Raises:
        ModelReplyError: the reply hit ``max_tokens`` with incomplete content, or does not
            parse.
    """
    content = [{"type": "image_url", "image_url": {"url": url}} for url in images]
    content.append({"type": "text", "text": prompt})
    body = {
        "model": endpoint["model_id"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": True},
        },
    }
    request = urllib.request.Request(
        endpoint["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
    )
    with urllib.request.urlopen(request, timeout=endpoint["request_timeout_s"]) as response:
        choice = json.loads(response.read())["choices"][0]
    raw = choice["message"]["content"]
    if choice["finish_reason"] == "length":
        recovered = close_stalled_reply(raw, schema)
        if recovered is None:
            raise ModelReplyError(f"reply cut off at {max_tokens} tokens", raw)
        return recovered, raw.rstrip() + " [closed]"
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as error:
        raise ModelReplyError(f"reply is not valid JSON ({error})", raw) from error
