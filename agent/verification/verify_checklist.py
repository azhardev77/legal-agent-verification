"""Does the agent's checklist cover what the case needs? run_verification is not implemented yet.

Called when the agent calls set_checklist with VERIFICATION=on, and by
./helper verify RUN --checklist.

    run_verification(ctx, items) -> {"passed": bool, "score": float, "feedback": str}

- items: the agent's checklist items: [{"id", "question", "evidence_needed", ...}, ...]
- load_reference(ctx.root) (pre-built, below) returns the case's ground truth,
  [{"id": "...", "question": "..."}, ...], or None when the case has none (then the
  checklist passes).

1. Ask Muse Spark (ctx.client.responses.create, JSON output) to mark each reference item
   covered or not by some agent item, even if worded differently, and to write feedback.
2. score = covered reference items / all reference items; passed = every one is covered.
3. feedback: what kind of question the checklist is missing, and any item that only
   restates background. It goes to the agent, so it must point to the direction only:
   never quote, paraphrase, or count the reference items. Empty when it passes.

Only passed and feedback reach the agent; the score is shown in the console.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from agent_kit.settings import case_dir, model
from openai.types.responses import Response

# ---- pre-built: the ground truth (don't change) ----------------------------------------
# It lives outside this folder (GROUND_TRUTH_DIR, default ../ground_truth), one
# <case folder name>.json per case. Read it only through load_reference.


def reference_path(root: Path) -> Path:
    folder = Path(os.environ.get("GROUND_TRUTH_DIR") or "../ground_truth").expanduser()
    if not folder.is_absolute():
        folder = (Path(root) / folder).resolve()
    return folder / f"{case_dir(root).name}.json"


def load_reference(root: Path) -> list | None:
    """The case's ground-truth items, or None if the case has none."""
    path = reference_path(root)
    return json.loads(path.read_text())["items"] if path.is_file() else None


# ---- yours ---------------------------------------------------------------------------

PROMPT = """REFERENCE CHECKLIST. The ground-truth questions this case turns on:
{reference}

The agent's checklist:
{agent}

For each ground-truth item, decide whether some agent item covers the same question, even if worded differently.
Answer only with JSON: {{"covered": {{"<ground-truth id>": true or false}}, "feedback": "..."}}.
The feedback goes to the agent when any ground-truth item is not covered. It must clearly instruct the agent on what kinds of questions are missing, pointing specifically to:
- Sequence and timeline of events leading up to go-live and deployment.
- Personnel-level responsibility, signing authority, and document authenticity.
- Guide the agent away from redundant points that only restate background already given in the assignment.
Point in the direction only: never quote, paraphrase, or count the ground-truth items. Empty feedback when every item is covered."""


def run_verification(ctx, items: list) -> dict:
    reference = load_reference(ctx.root)
    if reference is None:
        return {"passed": True, "score": None, "feedback": ""}
    if not reference:
        return {"passed": True, "score": 1.0, "feedback": ""}
    agent = [
        {
            "id": item.get("id"),
            "question": item.get("question", ""),
            "evidence_needed": item.get("evidence_needed", ""),
        }
        for item in items
        if isinstance(item, dict)
    ]
    response = cast(
        Response,
        ctx.client.responses.create(
            model=model(),
            input=PROMPT.format(
                reference=json.dumps(reference), agent=json.dumps(agent)
            ),
            text={"format": {"type": "json_object"}},
        ),
    )
    try:
        parsed = json.loads(response.output_text)
    except (ValueError, AttributeError):
        parsed = {}
    covered = parsed.get("covered") or {}
    if isinstance(covered, list):
        covered = {str(entry): True for entry in covered}
    hits = sum(1 for ref in reference if covered.get(ref.get("id")))
    passed = hits == len(reference)
    return {
        "passed": passed,
        "score": hits / len(reference),
        "feedback": "" if passed else str(parsed.get("feedback") or ""),
    }
