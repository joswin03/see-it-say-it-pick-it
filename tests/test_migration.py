"""CPU regression checks for migrated planning, schema recovery and baseline transport."""

import io
import json
import unittest
from unittest.mock import patch

from src.apertus import REQUEST_OPTIONS, Apertus
from src.decision import answer_instruction, decision_schema
from src.plans import prepare_moves
from src.structured import close_stalled_reply

PLACES = ["grey_tray", "black_tray", "aside"]
CONFIG = {
    "places": PLACES,
    "decision": {"enforce_clear_top": True, "max_attempts": 2},
    "gripper_max_opening_mm": 85,
}
OBJECTS = [
    {"id": "base", "pixel": [10, 10], "too_wide": False},
    {"id": "middle", "pixel": [20, 20], "too_wide": False},
    {"id": "top", "pixel": [30, 30], "too_wide": False},
    {"id": "wide", "pixel": [40, 40], "too_wide": True},
]


class MigrationTests(unittest.TestCase):
    """Check executable ordering and reject fabricated or unsafe plans."""

    def test_recursive_stack_clearing(self) -> None:
        """Clear a three-object stack from the top down.

        Args:
            None.

        Returns:
            None.
        """
        steps, error = prepare_moves(
            [{"object": "base", "destination": "hold"}],
            OBJECTS,
            [("top", "middle"), ("middle", "base")],
            CONFIG,
        )
        self.assertIsNone(error)
        self.assertEqual(
            [(step["action"], step["object"]) for step in steps],
            [
                ("pick", "top"),
                ("place", "aside"),
                ("pick", "middle"),
                ("place", "aside"),
                ("pick", "base"),
            ],
        )

    def test_invalid_plans_are_rejected_without_partial_moves(self) -> None:
        """Reject self-placement, unknown ids, wide blockers, held objects and stack cycles.

        Args:
            None.

        Returns:
            None.
        """
        cases = [
            ([{"object": "base", "destination": "base"}], []),
            ([{"object": "missing", "destination": "hold"}], []),
            ([{"object": "base", "destination": "hold"}], [("wide", "base")]),
            (
                [
                    {"object": "base", "destination": "hold"},
                    {"object": "top", "destination": "aside"},
                ],
                [],
            ),
            ([{"object": "base", "destination": "hold"}], [("base", "middle"), ("middle", "base")]),
        ]
        for moves, stacks in cases:
            with self.subTest(moves=moves, stacks=stacks):
                steps, error = prepare_moves(moves, OBJECTS, stacks, CONFIG)
                self.assertEqual(steps, [])
                self.assertIsNotNone(error)

    def test_placing_an_object_creates_a_new_blocker(self) -> None:
        """Track new stacks created by earlier moves in the same plan.

        Args:
            None.

        Returns:
            None.
        """
        steps, error = prepare_moves(
            [
                {"object": "top", "destination": "base"},
                {"object": "base", "destination": "hold"},
            ],
            OBJECTS,
            [],
            CONFIG,
        )
        self.assertIsNone(error)
        self.assertEqual(
            steps[2:4],
            [{"action": "pick", "object": "top"}, {"action": "place", "object": "aside"}],
        )

    def test_hold_before_a_later_move_of_the_same_object_is_dropped(self) -> None:
        """Keep the later destination when an object is first held and moved again later.

        Args:
            None.

        Returns:
            None.
        """
        steps, error = prepare_moves(
            [
                {"object": "base", "destination": "hold"},
                {"object": "top", "destination": "aside"},
                {"object": "base", "destination": "grey_tray"},
            ],
            OBJECTS,
            [],
            CONFIG,
        )
        self.assertIsNone(error)
        self.assertEqual([step["object"] for step in steps], ["top", "aside", "base", "grey_tray"])

    def test_plan_moving_a_kept_object_is_refused(self) -> None:
        """Refuse when clearing a stack must move an object the instruction keeps in place.

        Args:
            None.

        Returns:
            None.
        """
        config = {**CONFIG, "safety": {"check_keep_in_place": True}}
        plan = {"decision": "plan", "moves": [{"object": "base", "destination": "hold"}]}
        with patch("src.decision.decide", return_value=(plan, "plan")):
            result, _ = answer_instruction(
                "synthetic",
                "Lift the bottom object",
                "scene",
                OBJECTS,
                [("top", "base")],
                None,
                None,
                config,
                ["top"],
            )
        self.assertEqual(result["actions"][0]["action"], "refuse")
        self.assertIn("top", result["actions"][0]["reason"])

    def test_recovery_checks_the_entire_schema(self) -> None:
        """Close complete JSON without accepting unknown ids or incomplete fields.

        Args:
            None.

        Returns:
            None.
        """
        schema = decision_schema(["base"], 100, 3, PLACES)
        good = {
            "analysis": "Ready",
            "decision": "plan",
            "message": "",
            "moves": [{"object": "base", "destination": "hold"}],
        }
        self.assertEqual(close_stalled_reply(json.dumps(good)[:-1], schema), good)
        good["moves"][0]["object"] = "invented"
        self.assertIsNone(close_stalled_reply(json.dumps(good)[:-1], schema))
        self.assertIsNone(close_stalled_reply('{"analysis": "unfinished', schema))

    def test_plan_retry_recomputes_first_pick(self) -> None:
        """Give a rejected plan back to Apertus and target the first blocker in its correction.

        Args:
            None.

        Returns:
            None.
        """
        bad = {"decision": "plan", "moves": [{"object": "base", "destination": "base"}]}
        good = {"decision": "plan", "moves": [{"object": "base", "destination": "hold"}]}
        with patch("src.decision.decide", side_effect=[(bad, "bad"), (good, "good")]) as decide:
            result, attempts = answer_instruction(
                "synthetic",
                "Lift the bottom object",
                "scene",
                OBJECTS,
                [("top", "base")],
                None,
                None,
                CONFIG,
            )
        self.assertEqual(result["target_pixel"], [30, 30])
        self.assertIn("onto itself", decide.call_args.args[-1])
        self.assertEqual(len(attempts), 2)

    def test_schema_options_flow_through_unchanged_baseline(self) -> None:
        """Inject decoding options on baseline requests and reset them after each call.

        Args:
            None.

        Returns:
            None.
        """
        schema = {
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        }
        response = json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]}).encode()
        config = {
            "base_url": "https://api.inference.cscs.ch/v1",
            "api_key_env": "MIGRATION_TEST_KEY",
            "model_id": "test-model",
            "request_timeout_s": 1,
            "decode": {"temperature": 0},
        }
        with (
            patch.dict("os.environ", {"MIGRATION_TEST_KEY": "test-key"}),
            patch("urllib.request.urlopen", return_value=io.BytesIO(response)) as transport,
        ):
            model = Apertus(config, mock=False)
            reply, _ = model.ask_json("safety", "Check handling", schema, 123)
            request = transport.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(reply, {"ok": True})
            self.assertEqual(body["max_tokens"], 123)
            self.assertEqual(body["response_format"]["json_schema"]["schema"], schema)
            self.assertEqual([part["type"] for part in body["messages"][0]["content"]], ["text"])
            self.assertIsNone(REQUEST_OPTIONS.get())

    def test_screen_failures_stop_before_planning(self) -> None:
        """Fail explicitly when both safety replies are unusable.

        Args:
            None.

        Returns:
            None.
        """
        from pathlib import Path

        import numpy as np
        import yaml

        from src.graph import build_graph
        from src.structured import ModelReplyError

        config = yaml.safe_load(Path("configs/pipeline.yaml").read_text())
        objects = [
            {
                **item,
                "number": index,
                "label": item["id"],
                "size_mm": [30, 20],
                "table_xy": item["pixel"],
                "tray": "grey_tray",
                "is_hand": False,
            }
            for index, item in enumerate(OBJECTS, 1)
        ]
        graph = build_graph(config, Apertus(config, mock=True))
        with (
            patch(
                "src.graph.check_instruction", side_effect=ModelReplyError("bad safety reply", "{")
            ),
            patch("src.graph.answer_instruction") as planner,
        ):
            state = graph.invoke(
                {
                    "instruction_id": "synthetic",
                    "instruction": "Lift the block",
                    "marks": objects,
                    "on_top": [],
                    "annotated": np.zeros((10, 10, 3), np.uint8),
                    "result": None,
                }
            )
        planner.assert_not_called()
        self.assertEqual(state["result"]["actions"][0]["action"], "ask_user")
        self.assertEqual(len(state["attempts"]), config["decision"]["max_attempts"])

    def test_spatial_ranking_excludes_its_anchor(self) -> None:
        """Render image order and calibrated distances without ranking an object beside itself.

        Args:
            None.

        Returns:
            None.
        """
        from src.scene_graph import describe

        objects = [
            {
                "id": name,
                "number": index,
                "label": "blue block",
                "size_mm": [30, 20],
                "table_xy": [distance, 0],
                "pixel": [index * 10, 10],
                "tray": "grey_tray",
                "too_wide": False,
                "is_hand": False,
                "aliases": [],
            }
            for index, (name, distance) in enumerate([("a", 0), ("b", 30), ("c", 5)], 1)
        ]
        description = describe(objects, [("c", "a")])
        self.assertIn("LEFT TO RIGHT: a, b, c", description)
        self.assertIn("  a: c, b", description)
        self.assertIn("c lies on a", description)
