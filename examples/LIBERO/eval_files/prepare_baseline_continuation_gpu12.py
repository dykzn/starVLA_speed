#!/usr/bin/env python3
"""Prepare a duplicate-free GPU1/GPU2 continuation of the baseline run.

The original baseline was partly evaluated on GPU1/2 and then unfinished
shards were migrated to GPU0/3.  This script merges the migration logs back
into the migration-time manifests and emits one client job per unfinished
task, so completed episodes are never evaluated twice.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


FIELDS = [
    "variant",
    "gpu",
    "source_gpu",
    "suite",
    "shard",
    "task_id",
    "task_start",
    "task_end",
    "port",
    "output_dir",
    "log_file",
    "mpl_dir",
    "container_name",
    "screen_name",
    "target_episodes",
    "resume_manifest",
    "resume_completed",
    "resume_successes",
    "remaining_episodes",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--migration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = args.output_dir / "resume_specs"
    specs_dir.mkdir()

    # Reuse the parser that understands both the old and the current log
    # format, including Resume task_id markers.
    sys.path.insert(0, str(args.baseline_run_dir))
    from build_current_resume_specs import parse_current_log  # noqa: PLC0415

    baseline_rows = read_rows(args.baseline_run_dir / "jobs.tsv")
    migration_jobs_path = args.migration_dir / "jobs.tsv"
    migration_rows = read_rows(migration_jobs_path) if migration_jobs_path.is_file() else []
    migration_by_key = {
        (row["suite"], row["shard"]): row for row in migration_rows
    }

    trials = 50
    pending: list[dict[str, object]] = []
    all_tasks: dict[str, dict[str, object]] = {}
    cumulative_completed = 0
    cumulative_successes = 0

    for source in baseline_rows:
        if source.get("variant") == "variant":
            continue
        if source.get("variant") != "neutral":
            raise RuntimeError(f"unexpected baseline variant: {source}")
        suite = source["suite"]
        shard = source["shard"]
        key = (suite, shard)
        base_manifest = (
            args.migration_dir
            / "resume_specs"
            / f"gpu{source['gpu']}_{suite}_s{shard}.json"
        )
        if not base_manifest.is_file():
            raise FileNotFoundError(base_manifest)
        base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))

        migration_row = migration_by_key.get(key)
        if migration_row is not None:
            migration_log = Path(migration_row["log_file"])
            tasks, descriptions, completed, successes, _, _ = parse_current_log(
                migration_log,
                base_payload,
                int(source["task_start"]),
                int(source["task_end"]),
            )
        else:
            tasks = {
                str(task_id): dict(info)
                for task_id, info in base_payload.get("tasks", {}).items()
            }
            descriptions = {
                str(task_id): str(description)
                for task_id, description in base_payload.get(
                    "task_descriptions", {}
                ).items()
            }
            completed = int(base_payload.get("completed_total", 0))
            successes = int(base_payload.get("successes_total", 0))

        cumulative_completed += completed
        cumulative_successes += successes
        for task_id in range(int(source["task_start"]), int(source["task_end"])):
            info = tasks.get(str(task_id), {"completed": 0, "successes": 0})
            task_completed = int(info.get("completed", 0))
            task_successes = int(info.get("successes", 0))
            if not (0 <= task_successes <= task_completed <= trials):
                raise RuntimeError(
                    f"invalid cumulative counts for {suite}/task{task_id}: "
                    f"{task_completed}/{task_successes}"
                )
            task_key = f"{suite}/task{task_id}"
            all_tasks[task_key] = {
                "suite": suite,
                "task_id": task_id,
                "source_gpu": int(source["gpu"]),
                "source_shard": int(shard),
                "completed": task_completed,
                "successes": task_successes,
                "description": descriptions.get(str(task_id), ""),
            }
            remaining = trials - task_completed
            if remaining <= 0:
                continue

            manifest_payload = {
                "format": "starvla_resume_manifest_v1",
                "source_log": str(
                    Path(migration_row["log_file"])
                    if migration_row is not None
                    else base_payload.get("source_log", "")
                ),
                "base_manifest": str(base_manifest),
                "gpu": int(source["gpu"]),
                "suite": suite,
                "shard": int(shard),
                "task_start": task_id,
                "task_end": task_id + 1,
                "num_trials_per_task": trials,
                "tasks": {
                    str(task_id): {
                        "completed": task_completed,
                        "successes": task_successes,
                    }
                },
                "task_descriptions": {str(task_id): descriptions.get(str(task_id), "")},
                "completed_total": task_completed,
                "successes_total": task_successes,
            }
            manifest_path = specs_dir / f"{suite}_task{task_id}.json"
            write_json(manifest_path, manifest_payload)
            pending.append(
                {
                    "suite": suite,
                    "shard": int(shard),
                    "task_id": task_id,
                    "source_gpu": int(source["gpu"]),
                    "resume_manifest": str(manifest_path),
                    "resume_completed": task_completed,
                    "resume_successes": task_successes,
                    "remaining_episodes": remaining,
                }
            )

    if not pending:
        raise RuntimeError("all baseline tasks are already complete")

    # Greedily balance unfinished work while enforcing the user's 8-client
    # limit on each GPU.
    loads = {1: 0, 2: 0}
    counts = {1: 0, 2: 0}
    for item in sorted(
        pending,
        key=lambda value: (-int(value["remaining_episodes"]), str(value["suite"]), int(value["task_id"])),
    ):
        candidates = [gpu for gpu in (1, 2) if counts[gpu] < 8]
        if not candidates:
            raise RuntimeError(f"more than 16 pending tasks: {len(pending)}")
        gpu = min(candidates, key=lambda candidate: (loads[candidate], counts[candidate]))
        item["gpu"] = gpu
        loads[gpu] += int(item["remaining_episodes"])
        counts[gpu] += 1

    if any(count == 0 or count > 8 for count in counts.values()):
        raise RuntimeError(f"each GPU must have 1-8 clients, got {counts}")

    run_tag = datetime.now().strftime("%m%d%H%M%S")
    jobs_path = args.output_dir / "jobs.tsv"
    with jobs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="|", lineterminator="\n")
        writer.writeheader()
        for item in sorted(pending, key=lambda value: (int(value["gpu"]), str(value["suite"]), int(value["task_id"]))):
            gpu = int(item["gpu"])
            suite = str(item["suite"])
            task_id = int(item["task_id"])
            client_name = f"starvla_baseline_cont_g{gpu}_{suite}_t{task_id}_{run_tag}"
            output_dir = args.output_dir / "outputs" / f"gpu{gpu}" / suite / f"task{task_id}"
            log_file = args.output_dir / "logs" / f"gpu{gpu}_{suite}_task{task_id}.log"
            mpl_dir = args.output_dir / "mplconfig" / f"gpu{gpu}_{suite}_task{task_id}"
            writer.writerow(
                {
                    "variant": "neutral",
                    "gpu": gpu,
                    "source_gpu": item["source_gpu"],
                    "suite": suite,
                    "shard": item["shard"],
                    "task_id": task_id,
                    "task_start": task_id,
                    "task_end": task_id + 1,
                    "port": 6711 if gpu == 1 else 6712,
                    "output_dir": output_dir,
                    "log_file": log_file,
                    "mpl_dir": mpl_dir,
                    "container_name": client_name,
                    "screen_name": client_name,
                    "target_episodes": trials,
                    "resume_manifest": item["resume_manifest"],
                    "resume_completed": item["resume_completed"],
                    "resume_successes": item["resume_successes"],
                    "remaining_episodes": item["remaining_episodes"],
                }
            )

    state = {
        "source_baseline_run_dir": str(args.baseline_run_dir),
        "source_migration_dir": str(args.migration_dir),
        "baseline_completed_before_continuation": cumulative_completed,
        "baseline_successes_before_continuation": cumulative_successes,
        "baseline_target_episodes": sum(
            int(row["target_episodes"]) for row in baseline_rows if row.get("variant") == "neutral"
        ),
        "continuation_target_episodes": sum(
            int(item["remaining_episodes"]) for item in pending
        ),
        "pending_task_count": len(pending),
        "clients_per_gpu": counts,
        "episodes_per_gpu": loads,
        "tasks": all_tasks,
    }
    write_json(args.output_dir / "continuation_state.json", state)
    (args.output_dir / "RUN_INFO.md").write_text(
        "\n".join(
            [
                "# Baseline continuation on GPU1/GPU2",
                "",
                f"- Created: {datetime.now().astimezone().strftime('%F %T %Z')}",
                f"- Source baseline: {args.baseline_run_dir}",
                f"- Source migration: {args.migration_dir}",
                f"- GPU1 clients: {counts[1]} ({loads[1]} new episodes)",
                f"- GPU2 clients: {counts[2]} ({loads[2]} new episodes)",
                f"- Cumulative baseline before continuation: {cumulative_completed}/2000 episodes",
                f"- Remaining continuation target: {sum(int(item['remaining_episodes']) for item in pending)} episodes",
                "- Prompt variant: neutral (original task instruction)",
                "- Each client evaluates exactly one task and resumes from its cumulative manifest.",
                "",
                "Launch command:",
                f"  bash examples/LIBERO/eval_files/launch_baseline_continuation_gpu12.sh {args.output_dir}",
                "",
                "Monitor command:",
                f"  bash examples/LIBERO/eval_files/monitor_baseline_continuation_gpu12.sh {args.output_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(pending)} pending tasks; clients per GPU: {counts}; episodes: {loads}")
    print(f"cumulative before continuation: {cumulative_completed}/2000 episodes, {cumulative_successes} successes")
    print(f"jobs: {jobs_path}")


if __name__ == "__main__":
    main()
