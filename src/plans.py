"""Validate Joswin-style object/destination moves and clear blocking stacks.

Checks use measured scene facts only; Apertus interprets every instruction.
"""


def prepare_moves(
    moves: list[dict], marks: list[dict], on_top: list[tuple[str, str]], config: dict
) -> tuple[list[dict], str | None]:
    """Validate structured moves and clear blocking stacks before building an answer.

    Joswin's move format makes a pick and its destination inseparable. A "hold" move is dropped
    only when the same object is moved again later in the plan, so no destination is lost.
    An invalid plan returns feedback for Apertus to reconsider.

    Args:
        moves: Model-generated object and destination pairs.
        marks: Detected objects with graspability flags.
        on_top: Upper and lower object ids for the initial scene.
        config: Full pipeline configuration.

    Returns:
        Expanded steps and an error for retry, or None when executable.
    """
    by_id = {mark["id"]: mark for mark in marks}
    if not moves:
        return [], "a plan needs at least one move"
    kept = []
    for index, move in enumerate(moves):
        moved_later = any(later["object"] == move["object"] for later in moves[index + 1 :])
        if move["destination"] == "hold" and moved_later:
            continue
        kept.append(move)
    steps = []
    relations = {tuple(pair) for pair in on_top}

    def pick(object_id: str, visiting: set[str]) -> str | None:
        """Clear blockers recursively, then pick an object and update the simulated stack.

        Args:
            object_id: Object to pick next.
            visiting: Ancestors in the current traversal for cycle detection.

        Returns:
            A concrete problem, or None after appending executable steps.
        """
        if object_id in visiting:
            return "the perceived stacking relationships contain a cycle"
        item = by_id[object_id]
        if item["too_wide"]:
            return f"{object_id} is marked TOO WIDE TO GRIP"
        if item.get("is_hand", False):
            return f"{object_id} is a person's hand and cannot be picked"
        blockers = sorted(upper for upper, lower in relations if lower == object_id)
        if blockers and not config["decision"]["enforce_clear_top"]:
            return f"{object_id} is still covered by {', '.join(blockers)}"
        for upper in blockers:
            problem = pick(upper, visiting | {object_id})
            if problem:
                return problem
            steps.append({"action": "place", "object": "aside"})
        steps.append({"action": "pick", "object": object_id})
        relations.difference_update(pair for pair in list(relations) if pair[0] == object_id)
        return None

    for index, move in enumerate(kept):
        object_id, destination = move["object"], move["destination"]
        if object_id not in by_id:
            return [], f"{object_id} is not an object in the scene"
        if destination == object_id:
            return [], f"{object_id} cannot be moved onto itself"
        if destination not in config["places"] + ["hold"] and destination not in by_id:
            return [], f"{destination} is not a known destination"
        if destination in by_id and by_id[destination].get("is_hand", False):
            return [], "an object cannot be placed onto a person's hand"
        if destination == "hold" and index != len(kept) - 1:
            return [], (
                "only the last move can end with an object held: give every earlier object the "
                "destination the instruction names, or answer ask_user if the instruction asks "
                "for one object and several fit equally"
            )
        problem = pick(object_id, set())
        if problem:
            return [], problem
        if destination != "hold":
            steps.append({"action": "place", "object": destination})
            if destination in by_id:
                relations.add((object_id, destination))
    return steps, None
