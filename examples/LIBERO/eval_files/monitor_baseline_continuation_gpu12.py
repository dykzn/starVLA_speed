#!/usr/bin/env python3
"""Monitor the duplicate-free baseline continuation."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COMPLETED_RE = re.compile(r"# episodes completed so far:\s*(\d+)")
SUCCESS_TOTAL_RE = re.compile(r"# successes:\s*(\d+)\s+\(")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)\s*$", re.MULTILINE)


def screen_text() -> str:
    try:
        return subprocess.run(
            ["screen", "-ls"], check=False, capture_output=True, text=True
        ).stdout
    except OSError:
        return ""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} RUN_DIR")
    run_dir = Path(sys.argv[1])
    state = json.loads((run_dir / "continuation_state.json").read_text(encoding="utf-8"))
    with (run_dir / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="|"))
    screens = screen_text()

    print(f"[{datetime.now().astimezone().strftime('%F %T %Z')}] baseline continuation progress")
    print("GPU SUITE        TASK  CUMULATIVE/TARGET  NEW/TARGET  SUCCESSES  STATUS")
    print("----------------------------------------------------------------------------")
    new_completed_total = 0
    new_successes_total = 0
    for row in jobs:
        log_path = Path(row["log_file"])
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        completed_markers = COMPLETED_RE.findall(text)
        success_markers = SUCCESS_TOTAL_RE.findall(text)
        episode_results = SUCCESS_RE.findall(text)
        base_completed = int(row["resume_completed"])
        base_successes = int(row["resume_successes"])
        cumulative = int(completed_markers[-1]) if completed_markers else base_completed + len(episode_results)
        successes = int(success_markers[-1]) if success_markers else base_successes + sum(item == "True" for item in episode_results)
        cumulative = min(cumulative, int(row["target_episodes"]))
        new_completed = max(0, cumulative - base_completed)
        new_successes = max(0, successes - base_successes)
        new_completed_total += new_completed
        new_successes_total += new_successes
        if cumulative >= int(row["target_episodes"]):
            status = "DONE"
        elif f".{row['screen_name']}" in screens:
            status = "RUNNING"
        elif "Traceback (most recent call last)" in text or "ERROR:" in text:
            status = "FAILED/EXITED"
        else:
            status = "NOT_STARTED"
        print(
            f"{row['gpu']:>3} {row['suite']:<14} t{row['task_id']:<4} "
            f"{cumulative:>4}/{row['target_episodes']:<4}          "
            f"{new_completed:>3}/{row['remaining_episodes']:<3}      "
            f"{successes:>4}      {status}"
        )

    baseline_before = int(state["baseline_completed_before_continuation"])
    baseline_successes_before = int(state["baseline_successes_before_continuation"])
    baseline_target = int(state["baseline_target_episodes"])
    continuation_target = int(state["continuation_target_episodes"])
    print("----------------------------------------------------------------------------")
    print(
        f"Continuation: {new_completed_total}/{continuation_target} new episodes, "
        f"{new_successes_total} new successes"
    )
    print(
        f"Baseline cumulative: {baseline_before + new_completed_total}/{baseline_target} episodes, "
        f"{baseline_successes_before + new_successes_total} successes"
    )
    print(f"All episode targets reached: {'YES' if new_completed_total >= continuation_target else 'NO'}")


if __name__ == "__main__":
    main()
