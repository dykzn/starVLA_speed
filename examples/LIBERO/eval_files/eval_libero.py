import dataclasses
import json
import logging
import math
import os
import pathlib

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

PROMPT_PREFIXES = {
    # Keep the neutral setting byte-for-byte compatible with the original
    # task instruction. The four ESC variants are separated by valence and
    # arousal; all cues are prepended to the original task instruction.
    "neutral": "",
    "positive": "I am feeling calm and hopeful right now. ",
    "negative": "I am feeling sad and disappointed right now. ",
    "pos_high": "I am feeling excited and optimistic right now. ",
    "neg_high": "I am feeling tense and worried right now. ",
}


def _build_model_instruction(task_description: str, prompt_variant: str) -> str:
    try:
        prefix = PROMPT_PREFIXES[prompt_variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt_variant={prompt_variant!r}; "
            f"choose one of {sorted(PROMPT_PREFIXES)}"
        ) from exc
    return f"{prefix}{task_description}"


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_start: int = 0  # First task id to evaluate (inclusive), useful for parallel sharding.
    task_end: int = -1  # Last task id to evaluate (exclusive); -1 = end of suite.
    task_stride: int = 1  # Task id stride, useful for parallel sharding.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_videos: bool = False  # Video encoding is expensive; enable only for visualization.

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"
    prompt_variant: str = "neutral"  # neutral, positive, negative, pos_high, or neg_high
    resume_manifest: str = ""  # JSON task->completed/success counts from an earlier run
    noise_apply_interval: int = 1  # 1=every env step; 8=only before each 8-step action chunk


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )
    if args.noise_apply_interval < 1:
        raise ValueError("noise_apply_interval must be >= 1")
    logging.info(
        "Noise apply interval: %d env steps (policy action chunk size: %d)",
        args.noise_apply_interval,
        client_model.action_chunk_size,
    )
    if args.noise_apply_interval > 1 and args.noise_apply_interval != client_model.action_chunk_size:
        logging.warning(
            "noise_apply_interval=%d does not match action_chunk_size=%d; "
            "only use the fast setting when they are aligned",
            args.noise_apply_interval,
            client_model.action_chunk_size,
        )

    # Optional task range/stride allows several independent clients to evaluate
    # disjoint task shards concurrently without changing the default behavior.
    if args.task_start < 0 or args.task_start >= num_tasks_in_suite:
        raise ValueError(
            f"task_start={args.task_start} must be in [0, {num_tasks_in_suite})"
        )
    task_end = num_tasks_in_suite if args.task_end < 0 else min(args.task_end, num_tasks_in_suite)
    if task_end <= args.task_start:
        raise ValueError(f"task_end={task_end} must be greater than task_start={args.task_start}")
    if args.task_stride <= 0:
        raise ValueError(f"task_stride={args.task_stride} must be positive")

    task_ids = list(range(args.task_start, task_end, args.task_stride))
    # Preserve the original max_tasks semantics: cap the selected tasks from
    # the requested shard, while -1 keeps the full shard.
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]
    logging.info(
        f"Evaluating task ids {task_ids} of {num_tasks_in_suite} "
        f"(max_tasks={args.max_tasks}, task_start={args.task_start}, "
        f"task_end={args.task_end}, task_stride={args.task_stride})"
    )

    resume_tasks = {}
    if args.resume_manifest:
        with open(args.resume_manifest, "r", encoding="utf-8") as f:
            resume_payload = json.load(f)
        resume_tasks = resume_payload.get("tasks", resume_payload)
        if not isinstance(resume_tasks, dict):
            raise ValueError(
                f"resume_manifest={args.resume_manifest} must contain a task mapping"
            )
        logging.info("Resuming completed episodes from %s", args.resume_manifest)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)
        resume_info = resume_tasks.get(str(task_id), {})
        resume_episodes = int(resume_info.get("completed", 0))
        resume_successes = int(resume_info.get("successes", 0))
        if not (0 <= resume_successes <= resume_episodes <= args.num_trials_per_task):
            raise ValueError(
                f"Invalid resume counts for task_id={task_id}: "
                f"completed={resume_episodes}, successes={resume_successes}"
            )

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        model_instruction = _build_model_instruction(task_description, args.prompt_variant)

        # Start episodes
        task_episodes, task_successes = resume_episodes, resume_successes
        total_episodes += resume_episodes
        total_successes += resume_successes
        if resume_episodes:
            logging.info(
                "Resume task_id=%d: skipping %d completed episodes (%d successes); "
                "next episode index=%d",
                task_id, resume_episodes, resume_successes, resume_episodes,
            )
        for episode_idx in tqdm.tqdm(
            range(resume_episodes, args.num_trials_per_task),
            initial=resume_episodes,
            total=args.num_trials_per_task,
        ):
            # Record exactly the language instruction sent to the policy.
            logging.info(f"\nTask: {model_instruction}")

            # Reset environment
            client_model.reset(task_description=model_instruction)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = [] if args.save_videos else None

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    # Only the final stabilization observation is consumed by
                    # the first policy request in fast mode.
                    apply_noise = (
                        args.noise_apply_interval == 1
                        or t == args.num_steps_wait - 1
                    )
                    obs, reward, done, info = env.step(
                        LIBERO_DUMMY_ACTION, apply_noise=apply_noise
                    )
                    t += 1
                    continue

                # The server returns an action chunk.  Only the first step of
                # each chunk needs observations and a websocket round trip;
                # subsequent actions are already cached in the client.
                refresh_chunk = (
                    step % client_model.action_chunk_size == 0
                    or client_model.raw_actions is None
                )

                # IMPORTANT: rotate 180 degrees to match train preprocessing.
                # Keep the first camera only for optional video recording on
                # cached steps; avoid all image work when videos are disabled.
                if args.save_videos or refresh_chunk:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                else:
                    img = None
                if args.save_videos:
                    replay_images.append(img)

                if refresh_chunk:
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    example_dict = {
                        "image": [img, wrist_img],
                        "lang": model_instruction,
                    }
                    response = client_model.step(example=example_dict, step=step)
                else:
                    response = client_model.step(step=step)

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                # The returned observation is consumed only when the next
                # action chunk is requested.  Avoid running expensive visual
                # corruption for the cached steps in between.
                apply_noise = (
                    args.noise_apply_interval == 1
                    or (step + 1) % args.noise_apply_interval == 0
                )
                obs, reward, done, info = env.step(
                    delta_action.tolist(), apply_noise=apply_noise
                )
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Video encoding can block the simulator for seconds per episode;
            # keep it opt-in for fast evaluation.
            if args.save_videos:
                import imageio

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        # Release the MuJoCo/EGL context before constructing the next task's
        # environment. This matters when a long run evaluates many tasks in
        # one client process.
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    client_model.close()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        # LIBERO-plus's environment wrapper checks this value as a string.
        # The original LIBERO wrapper accepts pathlib.Path, but the plus
        # wrapper performs substring checks before normalizing the path.
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
