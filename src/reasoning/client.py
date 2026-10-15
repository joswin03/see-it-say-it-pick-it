"""Minimal client for an OpenAI-compatible chat endpoint (vLLM serving Apertus).

Replies are constrained to a JSON schema through ``response_format``, so every answer parses.
"""

import base64
import json
import urllib.request

import cv2
import numpy as np


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
        The parsed reply and the raw reply text.
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
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(request, timeout=endpoint["request_timeout_s"]) as response:
        raw = json.loads(response.read())["choices"][0]["message"]["content"]
    return json.loads(raw), raw
