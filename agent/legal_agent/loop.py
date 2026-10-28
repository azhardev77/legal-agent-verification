"""The Legal Agent: the agentic loop.

run_agent works the case; resume_run continues a run that stopped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from agent_kit import tools
from agent_kit.progress import load_progress, new_run, resume, save_progress
from agent_kit.settings import case_dir, model

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "system_instructions.md"

NO_TOOL_NUDGE = (
    "Call exactly one tool per turn. You answered in plain text; "
    "pick the next tool call instead."
)

LAST_TURN = (
    "This is your last turn. File the memo now with submit_memo, "
    "using what you have."
)


def run_agent(*, root: Path, client: OpenAI, turns: int = 30) -> dict:
    root = Path(root)
    instructions = INSTRUCTIONS_PATH.read_text()
    objective = (case_dir(root) / "objective.md").read_text()
    run_id = new_run(root, instructions=instructions, objective=objective)
    messages: list = [{"role": "user", "content": objective}]
    return _loop(client, root, run_id, messages, instructions, turns)


def resume_run(*, root: Path, run_id: str, client: OpenAI, turns: int = 15) -> dict:
    root = Path(root)
    step = resume(root, run_id)
    if step["message"] is None:
        turn = load_progress(root, run_id)["turn"]
        return {
            "run_id": run_id,
            "status": step["status"],
            "turn": turn,
            "note": step["note"],
        }
    instructions = (root / "workspace/runs" / run_id / "instructions.md").read_text()
    messages = list(load_progress(root, run_id)["messages"])
    messages.append({"role": "user", "content": step["message"]})
    return _loop(client, root, run_id, messages, instructions, turns)


def _loop(
    client: OpenAI,
    root: Path,
    run_id: str,
    messages: list,
    instructions: str,
    turns: int,
) -> dict:
    definitions = tools.definitions(root, run_id)
    for index in range(max(0, turns)):
        if index == turns - 1:
            messages.append({"role": "user", "content": LAST_TURN})
        try:
            response = client.responses.create(
                model=model(),
                instructions=instructions,
                input=messages,
                tools=definitions,
            )
        except openai.APIError as error:
            save_progress(root, run_id, messages, status="paused")
            return {
                "run_id": run_id,
                "status": "paused",
                "turn": load_progress(root, run_id)["turn"],
                "note": str(error),
            }
        output = [_plain(item) for item in response.output]
        messages.extend(output)
        call = next(
            (
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ),
            None,
        )
        if call is None:
            messages.append({"role": "user", "content": NO_TOOL_NUDGE})
            save_progress(root, run_id, messages, response)
            continue
        result = tools.run(client, root, run_id, call)
        messages.append(
            {
                "type": "function_call_output",
                "call_id": call.get("call_id"),
                "output": json.dumps(result),
            }
        )
        status = result.get("status", "running")
        save_progress(root, run_id, messages, response, status=status)
        if status != "running":
            return {
                "run_id": run_id,
                "status": status,
                "turn": load_progress(root, run_id)["turn"],
            }
    save_progress(root, run_id, messages, status="paused")
    return {
        "run_id": run_id,
        "status": "paused",
        "turn": load_progress(root, run_id)["turn"],
        "note": "turn limit reached; resume to continue",
    }


def _plain(value: Any) -> Any:
    """SDK objects (such as the items of response.output) as plain JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
