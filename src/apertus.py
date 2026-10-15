"""Talk to Apertus through the organisers' baseline code (baseline/baseline.py).

Every call uses ``baseline.encode`` (resize and JPEG), ``baseline.ask`` (one chat request,
temperature 0) and ``baseline.parse`` (first JSON object in the reply). The only additions:
the API key replaces baseline.py's fixed ``Bearer none`` header, server errors (HTTP 5xx) are
retried with the identical request, and a mock mode answers without any model.
"""

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "baseline"))
sys.path.insert(0, str(REPO_DIR / "starter"))

import baseline

PLACEHOLDER_AUTHORIZATION = "Bearer none"
SERVER_ERROR_ATTEMPTS = 5
SERVER_ERROR_PAUSE_S = 10
FIRST_SERVER_ERROR = 500
MOCK_REPLIES = {
    "annotate": '{"objects": [{"label": "mock object", "box": [100, 100, 200, 200]}]}',
    "decide": '{"actions": [{"action": "pick", "object": 1}]}',
    "zoom": '{"cell": 5}',
}


def send_with_key(api_key: str):
    """Wrap ``urllib.request.urlopen``: real key instead of the placeholder, retry on 5xx.

    Args:
        api_key: key for the endpoint.

    Returns:
        Replacement for ``urllib.request.urlopen``.
    """
    original_urlopen = urllib.request.urlopen

    def urlopen(request, *args, **kwargs):
        """Swap the placeholder authorization header, then send, retrying server errors.

        Args:
            request: ``urllib.request.Request`` built by ``baseline.ask``.
            *args: passed through to ``urlopen``.
            **kwargs: passed through to ``urlopen``.

        Returns:
            The response of the original ``urlopen``.
        """
        if request.get_header("Authorization") == PLACEHOLDER_AUTHORIZATION:
            request.add_header("Authorization", f"Bearer {api_key}")
        for attempt in range(1, SERVER_ERROR_ATTEMPTS + 1):
            try:
                return original_urlopen(request, *args, **kwargs)
            except urllib.error.HTTPError as error:
                if error.code < FIRST_SERVER_ERROR or attempt == SERVER_ERROR_ATTEMPTS:
                    raise
                print(f"server error {error.code}, retry {attempt} of {SERVER_ERROR_ATTEMPTS - 1}")
                time.sleep(SERVER_ERROR_PAUSE_S)

    return urlopen


class Apertus:
    """One prompt plus one image in, the parsed JSON reply out."""

    def __init__(self, config: dict, mock: bool) -> None:
        """Prepare the endpoint from the config.

        Args:
            config: run config with ``base_url``, ``model_id``, ``api_key_env``,
                ``request_timeout_s`` and ``image_max_side_px``.
            mock: if True, no model is contacted and ``MOCK_REPLIES`` answer.

        Returns:
            None.
        """
        self.config = config
        self.mock = mock
        if not mock:
            urllib.request.urlopen = send_with_key(os.environ[config["api_key_env"]])

    def ask(
        self, kind: str, template: str, fields: dict, rgb: np.ndarray
    ) -> tuple[dict | None, str, float]:
        """Send one prompt with one image and parse the reply.

        Args:
            kind: ``"annotate"`` or ``"decide"``; selects the mock reply.
            template: prompt template; ``{w}`` and ``{h}`` are filled with the size the model
                sees, every other placeholder from ``fields``.
            fields: values for the other placeholders.
            rgb: image to show, (H, W, 3) uint8 RGB, full resolution.

        Returns:
            The parsed reply (None when it holds no JSON), the raw reply text and the resize
            factor (resized size divided by full size).
        """
        image_b64, width, height, scale = baseline.encode(rgb, self.config["image_max_side_px"])
        prompt = template.format(w=width, h=height, **fields)
        if self.mock:
            raw = MOCK_REPLIES[kind]
        else:
            raw = baseline.ask(
                self.config["base_url"],
                self.config["model_id"],
                image_b64,
                prompt,
                self.config["request_timeout_s"],
            )
        return baseline.parse(raw or ""), raw or "", scale
