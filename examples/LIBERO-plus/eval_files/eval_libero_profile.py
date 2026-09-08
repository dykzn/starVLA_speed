#!/usr/bin/env python
"""Profile eval_libero: measure per-component timing for each episode."""
import dataclasses, json, logging, math, os, pathlib, sys, time

_LIBERO_PLUS_HOME = os.environ.get("LIBERO_HOME", "/data3/dengyongkang/my_project/LIBERO-plus")
if _LIBERO_PLUS_HOME not in sys.path:
    sys.path.insert(0, _LIBERO_PLUS_HOME)

import imageio, numpy as np, tqdm, tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val):
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 9884
    resize_size: list = dataclasses.field(default_factory=lambda: [224, 224])
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    start_idx: int = 0
    end_idx: int = 10  # only 10 tasks for profiling
    video_out_path: str = "results/libero_plus_eval/profile_videos"
    log_path: str = "results/libero_plus_eval/logs"
    seed: int = 7


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def profile_eval(args: Args) -> None:
    logging.info(f"Profiling {args.task_suite_name}, tasks [{args.start_idx}, {args.end_idx})")

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_suite_name == "libero_spatial": max_steps = 220
    elif args.task_suite_name == "libero_object": max_steps = 280
    elif args.task_suite_name == "libero_goal": max_steps = 300
    elif args.task_suite_name == "libero_10": max_steps = 520
    elif args.task_suite_name == "libero_90": max_steps = 400
    else: raise ValueError(f"Unknown suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client_model = ModelClient(host=args.host, port=args.port, image_size=args.resize_size)

    # Accumulators
    all_timings = []
    t_env_total = 0.0
    t_infer_total = 0.0
    t_video_total = 0.0
    t_preproc_total = 0.0
    t_warmup_total = 0.0

    start = max(0, args.start_idx)
    end = num_tasks_in_suite if args.end_idx < 0 else min(args.end_idx, num_tasks_in_suite)

    for task_id in range(start, end):
        t0 = time.perf_counter()

        # --- ENV CREATION ---
        t_env_start = time.perf_counter()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        t_env_create = time.perf_counter() - t_env_start
        t_env_total += t_env_create

        for episode_idx in range(args.num_trials_per_task):
            ep_infer_times = []
            ep_env_times = []
            ep_preproc_times = []
            ep_warmup_times = []

            client_model.reset(task_description=task_description)
            t_env_start = time.perf_counter()
            env.reset()
            t_env_reset = time.perf_counter() - t_env_start

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            step = 0
            replay_images = []
            episode_start = time.perf_counter()

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    t_ws = time.perf_counter()
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    ep_warmup_times.append(time.perf_counter() - t_ws)
                    t += 1
                    continue

                # --- PREPROC ---
                t_pp = time.perf_counter()
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_images.append(img)
                state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                ep_preproc_times.append(time.perf_counter() - t_pp)

                example_dict = {
                    "image": [img, wrist_img],
                    "lang": str(task_description),
                }

                # --- MODEL INFERENCE ---
                t_inf = time.perf_counter()
                response = client_model.step(example=example_dict, step=step)
                ep_infer_times.append(time.perf_counter() - t_inf)

                # --- PARSE ACTION ---
                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(f"Invalid action sizes")
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                # --- ENV STEP ---
                t_es = time.perf_counter()
                obs, reward, done, info = env.step(delta_action.tolist())
                ep_env_times.append(time.perf_counter() - t_es)

                if done:
                    break
                t += 1
                step += 1

            episode_time = time.perf_counter() - episode_start

            # --- VIDEO SAVE ---
            t_vs = time.perf_counter()
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:80]
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"profile_{task_segment}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=25,
            )
            t_video = time.perf_counter() - t_vs
            t_video_total += t_video

            # Aggregate this episode
            n_infer_calls = len(ep_infer_times)
            n_env_steps = len(ep_env_times)
            total_infer = sum(ep_infer_times)
            total_env = sum(ep_env_times)
            total_preproc = sum(ep_preproc_times)
            total_warmup = sum(ep_warmup_times)

            timing = {
                "task_id": task_id,
                "episode": episode_idx,
                "total_steps": t,
                "n_infer_calls": n_infer_calls,
                "n_env_steps": n_env_steps,
                "episode_time": episode_time,
                "env_create": t_env_create,
                "env_reset": t_env_reset,
                "warmup_total": total_warmup,
                "preproc_total": total_preproc,
                "inference_total": total_infer,
                "inference_avg": total_infer / n_infer_calls if n_infer_calls else 0,
                "env_step_total": total_env,
                "env_step_avg": total_env / n_env_steps if n_env_steps else 0,
                "video_save": t_video,
                "done": done,
            }
            all_timings.append(timing)

            # Print per-episode breakdown
            pct_infer = total_infer / episode_time * 100
            pct_env = total_env / episode_time * 100
            pct_preproc = total_preproc / episode_time * 100
            pct_warmup = total_warmup / episode_time * 100
            pct_video = t_video / episode_time * 100
            pct_other = 100 - pct_infer - pct_env - pct_preproc - pct_warmup - pct_video

            logging.info(
                f"⏱️  Episode {episode_idx} | {t} steps | {episode_time:.1f}s total | "
                f"infer={total_infer:.1f}s ({pct_infer:.0f}%) "
                f"env={total_env:.1f}s ({pct_env:.0f}%) "
                f"preproc={total_preproc:.1f}s ({pct_preproc:.0f}%) "
                f"warmup={total_warmup:.1f}s ({pct_warmup:.0f}%) "
                f"video={t_video:.1f}s ({pct_video:.0f}%) "
                f"other={pct_other:.0f}% | "
                f"infer_avg={total_infer/n_infer_calls:.2f}s/call env_avg={total_env/n_env_steps*1000:.0f}ms/step"
            )

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("PROFILE SUMMARY")
    print("=" * 80)
    episodes = len(all_timings)
    avg_ep = np.mean([t["episode_time"] for t in all_timings])
    avg_steps = np.mean([t["total_steps"] for t in all_timings])
    avg_infer = np.mean([t["inference_total"] for t in all_timings])
    avg_env = np.mean([t["env_step_total"] for t in all_timings])
    avg_preproc = np.mean([t["preproc_total"] for t in all_timings])
    avg_video = np.mean([t["video_save"] for t in all_timings])
    avg_env_create = np.mean([t["env_create"] for t in all_timings])
    avg_infer_call = np.mean([t["inference_avg"] for t in all_timings])
    avg_env_step = np.mean([t["env_step_avg"] for t in all_timings])

    print(f"  Episodes profiled:    {episodes}")
    print(f"  Avg episode time:     {avg_ep:.1f}s")
    print(f"  Avg steps/episode:    {avg_steps:.0f}")
    print(f"")
    print(f"  Per-episode breakdown:")
    print(f"    Inference:          {avg_infer:.1f}s ({avg_infer/avg_ep*100:.0f}%)  — {avg_infer_call:.2f}s/call")
    print(f"    Env simulation:     {avg_env:.1f}s ({avg_env/avg_ep*100:.0f}%)  — {avg_env_step*1000:.0f}ms/step")
    print(f"    Preprocessing:      {avg_preproc:.1f}s ({avg_preproc/avg_ep*100:.0f}%)")
    print(f"    Video save:         {avg_video:.1f}s ({avg_video/avg_ep*100:.0f}%)")
    print(f"    Env create/reset:   {avg_env_create:.1f}s")

    # Save detailed JSON
    out_path = os.path.join(args.log_path, f"profile_{args.task_suite_name}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {k: v for k, v in locals().items() if k.startswith("avg_")}, "episodes": all_timings}, f, indent=2)
    print(f"\n  Detailed timings saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s | %(message)s", datefmt="%m/%d [%H:%M:%S]", force=True)
    tyro.cli(profile_eval)
