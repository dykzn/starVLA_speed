#!/usr/bin/env python3
"""Monitor both LIBERO-plus variants running on GPU0 and GPU3."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


COMPLETED_RE = re.compile(r"# episodes completed so far:\s*(\d+)")
SUCCESS_TOTAL_RE = re.compile(r"# successes:\s*(\d+)\s+\(")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)\s*$", re.MULTILINE)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


def screen_text() -> str:
    try:
        return subprocess.run(
            ["screen", "-ls"], check=False, capture_output=True, text=True
        ).stdout
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    jobs = read_rows(args.run_dir / "jobs.tsv")
    screens = screen_text()

    print(f"[{datetime.now().astimezone().strftime('%F %T %Z')}] LIBERO-plus progress")
    print("VARIANT  GPU  SUITE        SHARD  TASKS/TARGET  SUCCESSES  STATUS")
    print("----------------------------------------------------------------")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    total_done = total_target = total_success = 0
    all_done = True
    for row in jobs:
        log_path = Path(row["log_file"])
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        completed_markers = COMPLETED_RE.findall(text)
        success_markers = SUCCESS_TOTAL_RE.findall(text)
        episode_results = SUCCESS_RE.findall(text)
        completed = int(completed_markers[-1]) if completed_markers else len(episode_results)
        successes = int(success_markers[-1]) if success_markers else sum(item == "True" for item in episode_results)
        target = int(row["target_episodes"])
        finished = f"finished LIBERO-plus GPU={row['gpu']} suite={row['suite']} shard={row['shard']}" in text
        if completed >= target or finished:
            status = "DONE"
        elif f".{row['screen_name']}" in screens:
            status = "RUNNING"
        elif "Traceback (most recent call last)" in text or "ERROR:" in text:
            status = "FAILED/EXITED"
        elif not log_path.is_file():
            status = "NOT_STARTED"
        else:
            status = "EXITED/STARTING"
        print(
            f"{row['variant']:<8} {row['gpu']:>3} {row['suite']:<12} "
            f"s{row['shard']:<5} {completed:>5}/{target:<5}      "
            f"{successes:>5}      {status}"
        )
        totals[row["variant"]][0] += min(completed, target)
        totals[row["variant"]][1] += target
        totals[row["variant"]][2] += successes
        total_done += min(completed, target)
        total_target += target
        total_success += successes
        all_done &= completed >= target or finished

    print("----------------------------------------------------------------")
    for variant in sorted(totals):
        done, target, successes = totals[variant]
        print(f"{variant}: {done}/{target} tasks, {successes} successes")
    print(f"Overall: {total_done}/{total_target} tasks, {total_success} successes")
    print(f"All task targets reached: {'YES' if all_done else 'NO'}")


if __name__ == "__main__":
    main()
