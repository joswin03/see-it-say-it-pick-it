"""Talk to Apertus through the organisers' baseline code (baseline/baseline.py).

Images use ``baseline.encode`` and every real call uses the unchanged ``baseline.ask``;
replies are validated against their JSON schema. The additions:
the API key (when the config's environment variable is set) replaces baseline.py's fixed
``Bearer none`` header, server errors (HTTP 5xx) are
retried, and scoped request options add schema constraints and token limits for the structured
stages.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from src.structured import ModelReplyError, close_stalled_reply

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "baseline"))
sys.path.insert(0, str(REPO_DIR / "starter"))

import baseline

REQUEST_OPTIONS: ContextVar[dict | None] = ContextVar("apertus_request_options", default=None)

PLACEHOLDER_AUTHORIZATION = "Bearer none"
SERVER_ERROR_ATTEMPTS = 5
SERVER_ERROR_PAUSE_S = 10
FIRST_SERVER_ERROR = 500


def send_with_key(api_key: str) -> Callable[..., Any]:
    """Adapt baseline requests with authentication, scoped decoding settings and retries.

    Args:
        api_key: key for the endpoint.

    Returns:
        Replacement for ``urllib.request.urlopen``.
    """
    original_urlopen = urllib.request.urlopen

    def urlopen(request: urllib.request.Request, *args: Any, **kwargs: Any) -> Any:
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
            options = REQUEST_OPTIONS.get()
            if options is not None:
                body = json.loads(request.data)
                body.update(options["decode"])
                if options["text_only"]:
                    body["messages"][0]["content"] = [
                        part for part in body["messages"][0]["content"] if part["type"] == "text"
                    ]
                request.data = json.dumps(body).encode()
                request.remove_header("Content-length")
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

    def __init__(self, config: dict, mock: bool, log_path: Path | None = None) -> None:
        """Prepare the endpoint from the config.

        Args:
            config: run config with ``base_url``, ``model_id``, ``api_key_env``,
                ``request_timeout_s`` and ``image_max_side_px``.
            mock: if True, no model is contacted and fixed schema-valid replies answer.
            log_path: Optional append-only record of model replies without credentials or images.

        Returns:
            None.
        """
        self.config = config
        self.mock = mock
        self.log_path = log_path
        if not mock:
            urllib.request.urlopen = send_with_key(os.environ.get(config["api_key_env"], "none"))

    def ask_json(
        self,
        kind: str,
        prompt: str,
        schema: dict,
        max_tokens: int,
        rgb: np.ndarray | None = None,
        max_side: int | None = None,
    ) -> tuple[dict, str]:
        """Ask Apertus with Joswin's schema constraint and validate the entire reply.

        Args:
            kind: Naming, safety or decision stage.
            prompt: Complete prompt without interpolation.
            schema: JSON schema for the reply.
            max_tokens: Generation limit from the config.
            rgb: Optional image; safety can reason from scene text alone.
            max_side: Optional image size override.

        Returns:
            Validated reply and its raw text.
        """
        encoded = ""
        if rgb is not None:
            encoded, *_ = baseline.encode(rgb, max_side or self.config["image_max_side_px"])
        if self.mock:
            mocks = {
                "name": {"name": "mock object", "other_names": [], "is_object": True},
                "safety": {
                    "action_words": "mock",
                    "analysis": "Mock screen",
                    "involves_person": False,
                    "manner": "pick and place",
                    "message": "",
                },
                "boxes": {"objects": [{"label": "mock object", "box": [100, 100, 300, 300]}]},
                "decision": {
                    "analysis": "Mock decision",
                    "decision": "ask_user",
                    "message": "Mock mode has no model.",
                    "moves": [],
                },
            }
            reply = mocks[kind]
            Draft202012Validator(schema).validate(reply)
            self.log_reply(kind, json.dumps(reply))
            return reply, json.dumps(reply)
        options = {
            "text_only": rgb is None,
            "decode": {
                "max_tokens": max_tokens,
                "temperature": self.config["decode"]["temperature"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema, "strict": True},
                },
            },
        }
        token = REQUEST_OPTIONS.set(options)
        try:
            raw = (
                baseline.ask(
                    self.config["base_url"],
                    self.config["model_id"],
                    encoded,
                    prompt,
                    self.config["request_timeout_s"],
                )
                or ""
            )
        finally:
            REQUEST_OPTIONS.reset(token)
        self.log_reply(kind, raw)
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError as error:
            reply = close_stalled_reply(raw, schema)
            if reply is None:
                raise ModelReplyError(f"{kind} reply is incomplete or invalid JSON", raw) from error
        errors = list(Draft202012Validator(schema).iter_errors(reply))
        if errors:
            raise ModelReplyError(f"{kind} reply violates its schema: {errors[0].message}", raw)
        return reply, raw

    def log_reply(self, kind: str, raw: str) -> None:
        """Append one reply to the run audit log.

        Args:
            kind: Model stage that generated this reply.
            raw: Original reply text.

        Returns:
            None.
        """
        if self.log_path is not None:
            with self.log_path.open("a") as log:
                log.write(json.dumps({"stage": kind, "raw": raw}) + "\n")
