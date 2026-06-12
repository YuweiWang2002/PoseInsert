"""Evaluate BiRelPoseDP on RoboTwin seeds with RoboTwin-style rollout.

This script follows RoboTwin's official eval pattern for each seed:
1. optionally run expert play_once as a feasibility check;
2. reset the same seed;
3. repeatedly build runtime rel_obs, sample an ee action sequence, execute a
   small number of actions, and check task success.

It does not train or collect demonstrations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

POSEINSERT_ROOT = Path(__file__).resolve().parents[1]
if str(POSEINSERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEINSERT_ROOT))

from robotwin_birelpose.inference import load_birelpose_policy, predict_action_sequence, sanitize_ee_action
from robotwin_birelpose.relative_action import relative_action_to_absolute_ee_action
from robotwin_birelpose.relation_graph import build_bimanual_relation_obs
from robotwin_birelpose.task_adapters import extract_object_poses_from_task


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin_root", required=True)
    parser.add_argument("--task_name", default="handover_block")
    parser.add_argument("--task_config", default="handover_block_pose_clean50")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed_list_file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--exec_steps", type=int, default=1)
    parser.add_argument("--rollout_mode", choices=["receding", "chunked", "open_loop"], default="receding")
    parser.add_argument("--output_mode", choices=["auto", "absolute", "relative"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object_index", type=int, default=0)
    parser.add_argument("--expert_check", action="store_true")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--debug_trace", action="store_true")
    parser.add_argument("--debug_trace_path", default=None)
    parser.add_argument("--debug_max_control_steps", type=int, default=None)
    parser.add_argument("--debug_manual_action_test", action="store_true")
    parser.add_argument("--debug_manual_delta", type=float, default=0.02)
    parser.add_argument("--training_data_dir", default=None)
    return parser.parse_args()


def resolve_output_mode(args, metadata: dict) -> str:
    """Resolve whether the model output is executable ee action or rel_action."""
    if args.output_mode != "auto":
        return args.output_mode
    config = metadata.get("config", {})
    action_key = config.get("action_key")
    if action_key == "rel_action" or int(metadata.get("action_dim", 0)) == 20:
        return "relative"
    return "absolute"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_task(robotwin_root: Path, task_name: str):
    sys.path.insert(0, str(robotwin_root))
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def prepare_task_args(robotwin_root: Path, task_name: str, task_config: str) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    args = load_yaml(robotwin_root / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_data"] = False
    args["collect_data"] = False
    args["render_freq"] = 0
    args["eval_mode"] = True
    args["data_type"]["rgb"] = False
    args["data_type"]["depth"] = False
    args["data_type"]["pointcloud"] = False
    args["data_type"]["endpose"] = True
    args["data_type"]["qpos"] = True

    embodiment_type = args["embodiment"]
    embodiment_types = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment file for {name!r}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise ValueError("embodiment config must contain 1 or 3 entries")

    args["left_embodiment_config"] = load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = load_yaml(Path(args["right_robot_file"]) / "config.yml")
    return args


def get_head_camera_video_size(robotwin_root: Path, task_args: dict) -> str:
    camera_config = load_yaml(robotwin_root / "task_config" / "_camera_config.yml")
    camera_type = task_args["camera"]["head_camera_type"]
    if camera_type not in camera_config:
        raise KeyError(f"Unknown head camera type {camera_type!r}")
    return f"{camera_config[camera_type]['w']}x{camera_config[camera_type]['h']}"


def load_seed_list(args) -> list[int]:
    if args.seed is not None:
        return [int(args.seed)]
    if args.seed_list_file is None:
        return list(range(args.seed_start, args.seed_start + args.num_episodes))
    text = Path(args.seed_list_file).read_text(encoding="utf-8")
    seeds = [int(x) for x in text.split()]
    return seeds[: args.num_episodes]


def arr_list(value) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def array_stats(value) -> dict:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "finite": bool(np.isfinite(arr).all()),
    }


def quat_angle(q_a, q_b) -> float:
    qa = np.asarray(q_a, dtype=np.float64)
    qb = np.asarray(q_b, dtype=np.float64)
    qa = qa / max(np.linalg.norm(qa), 1e-12)
    qb = qb / max(np.linalg.norm(qb), 1e-12)
    dot = float(abs(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * np.arccos(dot))


def get_runtime_state(task, task_name: str, object_index: int) -> dict:
    obs = task.get_obs()
    endpose = obs.get("endpose", {})
    object_result = extract_object_poses_from_task(task, task_name, object_index=object_index)
    state = {
        "left_endpose": np.asarray(endpose["left_endpose"], dtype=np.float32),
        "right_endpose": np.asarray(endpose["right_endpose"], dtype=np.float32),
        "left_gripper": float(endpose["left_gripper"]),
        "right_gripper": float(endpose["right_gripper"]),
        "object_pose": np.asarray(object_result["object_pose"], dtype=np.float32),
        "object_name": object_result["object_name"],
    }
    return state


def rel_obs_from_state(state: dict) -> np.ndarray:
    rel_obs = build_bimanual_relation_obs(
        left_ee_pose=state["left_endpose"],
        right_ee_pose=state["right_endpose"],
        object_pose=state["object_pose"],
        left_gripper=state["left_gripper"],
        right_gripper=state["right_gripper"],
    )
    if rel_obs.shape != (29,):
        raise ValueError(f"rel_obs must have shape (29,), got {rel_obs.shape}")
    return rel_obs


def state_to_json(state: dict) -> dict:
    return {
        "left_endpose": arr_list(state["left_endpose"]),
        "right_endpose": arr_list(state["right_endpose"]),
        "left_gripper": state["left_gripper"],
        "right_gripper": state["right_gripper"],
        "object_pose": arr_list(state["object_pose"]),
        "object_name": state["object_name"],
    }


@torch.no_grad()
def sample_action_with_debug(model, normalizer: dict, obs_queue, device) -> dict:
    obs_queue = np.asarray(obs_queue, dtype=np.float32)
    normalized_obs = (obs_queue - normalizer["obs_mean"]) / normalizer["obs_std"]
    obs_tensor = torch.as_tensor(normalized_obs[None], dtype=torch.float32, device=device)
    action_norm = model.sample_action(obs_tensor).detach().cpu().numpy()[0]
    action_denorm = action_norm * normalizer["action_std"] + normalizer["action_mean"]
    return {
        "obs_queue": obs_queue,
        "normalized_obs": normalized_obs,
        "action_norm": action_norm.astype(np.float32),
        "action_denorm": action_denorm.astype(np.float32),
    }


def action_pose_deltas(state: dict, action: np.ndarray) -> dict:
    return {
        "left_position_delta_norm": float(np.linalg.norm(action[:3] - state["left_endpose"][:3])),
        "right_position_delta_norm": float(np.linalg.norm(action[8:11] - state["right_endpose"][:3])),
        "left_quat_angle_rad": quat_angle(action[3:7], state["left_endpose"][3:7]),
        "right_quat_angle_rad": quat_angle(action[11:15], state["right_endpose"][3:7]),
        "left_gripper_diff": float(action[7] - state["left_gripper"]),
        "right_gripper_diff": float(action[15] - state["right_gripper"]),
    }


def action_seq_motion_stats(action_seq: np.ndarray) -> dict:
    left = np.asarray(action_seq[:, :3], dtype=np.float64)
    right = np.asarray(action_seq[:, 8:11], dtype=np.float64)
    left_steps = np.linalg.norm(np.diff(left, axis=0), axis=1)
    right_steps = np.linalg.norm(np.diff(right, axis=0), axis=1)
    return {
        "action_seq_left_xyz_delta_norms": left_steps.tolist(),
        "action_seq_right_xyz_delta_norms": right_steps.tolist(),
        "first_to_last_left_delta": float(np.linalg.norm(left[-1] - left[0])),
        "first_to_last_right_delta": float(np.linalg.norm(right[-1] - right[0])),
        "left_step_delta_mean": float(left_steps.mean()) if left_steps.size else 0.0,
        "left_step_delta_max": float(left_steps.max()) if left_steps.size else 0.0,
        "right_step_delta_mean": float(right_steps.mean()) if right_steps.size else 0.0,
        "right_step_delta_max": float(right_steps.max()) if right_steps.size else 0.0,
    }


def model_action_seq_motion_stats(action_seq: np.ndarray, output_mode: str) -> dict:
    """Motion stats in the model-output space for debug traces."""
    if output_mode == "relative":
        left = np.asarray(action_seq[:, 6:9], dtype=np.float64)
        right = np.asarray(action_seq[:, 15:18], dtype=np.float64)
    else:
        left = np.asarray(action_seq[:, :3], dtype=np.float64)
        right = np.asarray(action_seq[:, 8:11], dtype=np.float64)
    left_steps = np.linalg.norm(np.diff(left, axis=0), axis=1)
    right_steps = np.linalg.norm(np.diff(right, axis=0), axis=1)
    return {
        "output_mode": output_mode,
        "left_xyz_delta_norms": left_steps.tolist(),
        "right_xyz_delta_norms": right_steps.tolist(),
        "first_to_last_left_delta": float(np.linalg.norm(left[-1] - left[0])),
        "first_to_last_right_delta": float(np.linalg.norm(right[-1] - right[0])),
        "left_step_delta_mean": float(left_steps.mean()) if left_steps.size else 0.0,
        "left_step_delta_max": float(left_steps.max()) if left_steps.size else 0.0,
        "right_step_delta_mean": float(right_steps.mean()) if right_steps.size else 0.0,
        "right_step_delta_max": float(right_steps.max()) if right_steps.size else 0.0,
    }


def model_action_to_ee_action(model_action: np.ndarray, object_pose: np.ndarray, output_mode: str) -> np.ndarray:
    """Convert one model output action to RoboTwin executable ``ee`` action."""
    if output_mode == "absolute":
        return np.asarray(model_action, dtype=np.float32)
    if output_mode == "relative":
        return relative_action_to_absolute_ee_action(model_action, object_pose)
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def actual_movement(before: dict, after: dict) -> dict:
    return {
        "left_actual_movement_norm": float(np.linalg.norm(after["left_endpose"][:3] - before["left_endpose"][:3])),
        "right_actual_movement_norm": float(np.linalg.norm(after["right_endpose"][:3] - before["right_endpose"][:3])),
        "left_quat_angle_rad": quat_angle(after["left_endpose"][3:7], before["left_endpose"][3:7]),
        "right_quat_angle_rad": quat_angle(after["right_endpose"][3:7], before["right_endpose"][3:7]),
        "left_gripper_movement": float(after["left_gripper"] - before["left_gripper"]),
        "right_gripper_movement": float(after["right_gripper"] - before["right_gripper"]),
    }


def open_trace(path):
    if path is None:
        raise ValueError("--debug_trace_path is required when --debug_trace is set")
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return trace_path.open("w", encoding="utf-8")


def write_trace(trace_file, item: dict) -> None:
    if trace_file is not None:
        trace_file.write(json.dumps(item) + "\n")
        trace_file.flush()


def start_video(task, task_args: dict, video_dir, episode_id: int, seed: int):
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode{episode_id}_seed{seed}.mp4"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            task_args["video_size"],
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    task._set_eval_video_ffmpeg(ffmpeg)
    return ffmpeg, str(video_path)


def print_checkpoint_diagnostics(metadata: dict, normalizer: dict) -> None:
    print("--- checkpoint diagnostics ---")
    print(f"obs_dim: {metadata['obs_dim']}")
    print(f"action_dim: {metadata['action_dim']}")
    print(f"horizon: {metadata['horizon']}")
    print(f"n_obs_steps: {metadata['n_obs_steps']}")
    print(f"normalizer_keys: {sorted(normalizer.keys())}")
    print(f"obs_mean_shape: {normalizer['obs_mean'].shape}, first10: {normalizer['obs_mean'][:10].tolist()}")
    print(
        f"obs_std_shape: {normalizer['obs_std'].shape}, "
        f"min: {float(normalizer['obs_std'].min())}, max: {float(normalizer['obs_std'].max())}"
    )
    print(f"action_mean_shape: {normalizer['action_mean'].shape}, values: {normalizer['action_mean'].tolist()}")
    print(f"action_std_shape: {normalizer['action_std'].shape}, values: {normalizer['action_std'].tolist()}")
    print(f"action_std_min: {float(normalizer['action_std'].min())}")
    config = metadata.get("config", {})
    print(
        "training_config: "
        f"horizon={config.get('horizon')}, n_obs_steps={config.get('n_obs_steps')}, "
        f"obs_dim={config.get('obs_dim')}, action_dim={config.get('action_dim')}, "
        "target_relation=False"
    )


def print_training_action_distribution(data_dir) -> None:
    if data_dir is None:
        print("training_data_dir: None")
        return
    paths = sorted(Path(data_dir).glob("*.npz"))
    print("--- training action distribution ---")
    print(f"training_data_dir: {data_dir}")
    print(f"npz_count: {len(paths)}")
    if not paths:
        return
    actions = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            actions.append(np.asarray(data["action_ee"], dtype=np.float32))
    action = np.concatenate(actions, axis=0)
    quat_norm_left = np.linalg.norm(action[:, 3:7], axis=1)
    quat_norm_right = np.linalg.norm(action[:, 11:15], axis=1)
    print(f"action_ee shape: {action.shape}")
    print(f"action mean: {action.mean(axis=0).tolist()}")
    print(f"action std: {action.std(axis=0).tolist()}")
    print(f"action min: {action.min(axis=0).tolist()}")
    print(f"action max: {action.max(axis=0).tolist()}")
    print(f"left xyz range: min={action[:, :3].min(axis=0).tolist()}, max={action[:, :3].max(axis=0).tolist()}")
    print(f"right xyz range: min={action[:, 8:11].min(axis=0).tolist()}, max={action[:, 8:11].max(axis=0).tolist()}")
    print(f"gripper range: left=({float(action[:, 7].min())}, {float(action[:, 7].max())}), "
          f"right=({float(action[:, 15].min())}, {float(action[:, 15].max())})")
    print(f"quat norm range: left=({float(quat_norm_left.min())}, {float(quat_norm_left.max())}), "
          f"right=({float(quat_norm_right.min())}, {float(quat_norm_right.max())})")


def rel_obs_from_task(task, task_name: str, object_index: int) -> np.ndarray:
    return rel_obs_from_state(get_runtime_state(task, task_name, object_index))


def rollout_one_seed(task, task_name, task_args, model, normalizer, metadata, seed, episode_id, args) -> dict:
    result = {
        "episode_id": episode_id,
        "seed": seed,
        "expert_ok": None,
        "success": False,
        "steps": 0,
        "failure_reason": None,
        "video_path": None,
        "error": None,
    }

    if args.expert_check:
        try:
            task.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
            task.play_once()
            result["expert_ok"] = bool(task.plan_success and task.check_success())
        finally:
            task.close_env()
        if not result["expert_ok"]:
            result["failure_reason"] = "expert_check_failed"
            return result

    trace_file = open_trace(args.debug_trace_path) if args.debug_trace else None
    output_mode = resolve_output_mode(args, metadata)
    if output_mode == "absolute" and int(metadata["action_dim"]) != 16:
        raise ValueError(f"absolute output_mode requires action_dim=16, got {metadata['action_dim']}")
    if output_mode == "relative" and int(metadata["action_dim"]) != 20:
        raise ValueError(f"relative output_mode requires action_dim=20, got {metadata['action_dim']}")
    ffmpeg = None
    try:
        task.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
        if args.save_video:
            ffmpeg, video_path = start_video(task, task_args, args.video_dir, episode_id, seed)
            result["video_path"] = video_path

        first_state = get_runtime_state(task, task_name, args.object_index)
        first_rel_obs = rel_obs_from_state(first_state)
        obs_queue = deque(
            [first_rel_obs.copy() for _ in range(metadata["n_obs_steps"])],
            maxlen=metadata["n_obs_steps"],
        )
        max_steps = args.max_steps if args.max_steps is not None else int(task.step_lim)
        if args.debug_max_control_steps is not None:
            max_steps = min(max_steps, args.debug_max_control_steps)

        while task.take_action_cnt < task.step_lim and result["steps"] < max_steps:
            before_state = get_runtime_state(task, task_name, args.object_index)
            before_rel_obs = rel_obs_from_state(before_state)
            if len(obs_queue) == obs_queue.maxlen:
                obs_queue[-1] = before_rel_obs.copy()
            sample_debug = sample_action_with_debug(model, normalizer, np.stack(list(obs_queue), axis=0), args.device)
            action_seq = sample_debug["action_denorm"]
            if args.rollout_mode == "open_loop":
                actions_to_execute = min(action_seq.shape[0], max_steps - result["steps"])
            else:
                actions_to_execute = min(args.exec_steps, action_seq.shape[0], max_steps - result["steps"])
            model_seq_motion = model_action_seq_motion_stats(action_seq, output_mode)

            for i in range(actions_to_execute):
                before_action_state = get_runtime_state(task, task_name, args.object_index)
                model_action = np.asarray(action_seq[i], dtype=np.float32)
                action_ee = model_action_to_ee_action(model_action, before_action_state["object_pose"], output_mode)
                action = sanitize_ee_action(action_ee)
                target_delta = action_pose_deltas(before_action_state, action)
                take_action_return = task.take_action(action, action_type="ee")
                result["steps"] += 1
                after_state = get_runtime_state(task, task_name, args.object_index)
                movement = actual_movement(before_action_state, after_state)

                write_trace(
                    trace_file,
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "step": result["steps"] - 1,
                        "current_obs": state_to_json(before_action_state),
                        "rel_obs": {
                            **array_stats(before_rel_obs),
                            "first10": arr_list(before_rel_obs[:10]),
                        },
                        "obs_queue": {
                            **array_stats(sample_debug["obs_queue"]),
                            "two_frames_equal": bool(
                                sample_debug["obs_queue"].shape[0] >= 2
                                and np.allclose(sample_debug["obs_queue"][0], sample_debug["obs_queue"][-1])
                            ),
                        },
                        "normalized_obs": array_stats(sample_debug["normalized_obs"]),
                        "model_raw_normalized_action": {
                            **array_stats(sample_debug["action_norm"]),
                            "first_action": arr_list(sample_debug["action_norm"][0]),
                        },
                        "model_output_mode": output_mode,
                        "model_action_seq_motion": model_seq_motion,
                        "action_seq_motion": model_seq_motion,
                        "denormalized_action": {
                            **array_stats(sample_debug["action_denorm"]),
                            "first_action": arr_list(sample_debug["action_denorm"][0]),
                        },
                        "model_action": arr_list(model_action),
                        "sanitized_action": {
                            "action": arr_list(action),
                            "left_xyz": arr_list(action[:3]),
                            "left_quat_wxyz": arr_list(action[3:7]),
                            "left_gripper": float(action[7]),
                            "right_xyz": arr_list(action[8:11]),
                            "right_quat_wxyz": arr_list(action[11:15]),
                            "right_gripper": float(action[15]),
                        },
                        "target_delta": target_delta,
                        "before_left_endpose": arr_list(before_action_state["left_endpose"]),
                        "after_left_endpose": arr_list(after_state["left_endpose"]),
                        "before_right_endpose": arr_list(before_action_state["right_endpose"]),
                        "after_right_endpose": arr_list(after_state["right_endpose"]),
                        "actual_movement": movement,
                        "take_action_return": str(take_action_return),
                    },
                )

                if bool(getattr(task, "eval_success", False)) or bool(task.check_success()):
                    result["success"] = True
                    return result
                if task.take_action_cnt >= task.step_lim or result["steps"] >= max_steps:
                    break
                obs_queue.append(rel_obs_from_task(task, task_name, args.object_index))
            if args.rollout_mode == "open_loop":
                break
    except Exception as exc:
        result["error"] = repr(exc)
        result["failure_reason"] = "exception"
    finally:
        if ffmpeg is not None:
            task._del_eval_video_ffmpeg()
        if trace_file is not None:
            trace_file.close()
        task.close_env()

    if not result["success"] and result["failure_reason"] is None:
        if args.max_steps is not None and result["steps"] >= args.max_steps:
            result["failure_reason"] = "max_steps_reached"
        elif getattr(task, "take_action_cnt", 0) >= getattr(task, "step_lim", 0):
            result["failure_reason"] = "step_lim_reached"
        else:
            result["failure_reason"] = "not_success"
    return result


def run_manual_action_test(task, task_name, task_args, seed: int, episode_id: int, args) -> dict:
    result = {
        "episode_id": episode_id,
        "seed": seed,
        "success": False,
        "steps": 0,
        "failure_reason": None,
        "video_path": None,
        "error": None,
    }
    trace_file = open_trace(args.debug_trace_path) if args.debug_trace else None
    ffmpeg = None
    try:
        task.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
        if args.save_video:
            ffmpeg, video_path = start_video(task, task_args, args.video_dir, episode_id, seed)
            result["video_path"] = video_path

        before = get_runtime_state(task, task_name, args.object_index)
        action = np.concatenate(
            [
                before["left_endpose"][:3] + np.array([args.debug_manual_delta, 0.0, 0.0], dtype=np.float32),
                before["left_endpose"][3:7],
                np.array([before["left_gripper"]], dtype=np.float32),
                before["right_endpose"][:3] + np.array([0.0, 0.0, args.debug_manual_delta], dtype=np.float32),
                before["right_endpose"][3:7],
                np.array([before["right_gripper"]], dtype=np.float32),
            ]
        ).astype(np.float32)
        action = sanitize_ee_action(action)
        target_delta = action_pose_deltas(before, action)
        take_action_return = task.take_action(action, action_type="ee")
        result["steps"] = 1
        after = get_runtime_state(task, task_name, args.object_index)
        movement = actual_movement(before, after)
        result["success"] = bool(movement["left_actual_movement_norm"] > 1e-4 or movement["right_actual_movement_norm"] > 1e-4)
        result["failure_reason"] = None if result["success"] else "manual_action_no_movement"

        write_trace(
            trace_file,
            {
                "mode": "manual_action_test",
                "episode_id": episode_id,
                "seed": seed,
                "before": state_to_json(before),
                "target_action": arr_list(action),
                "target_delta": target_delta,
                "after": state_to_json(after),
                "actual_movement": movement,
                "take_action_return": str(take_action_return),
            },
        )
    except Exception as exc:
        result["error"] = repr(exc)
        result["failure_reason"] = "exception"
    finally:
        if ffmpeg is not None:
            task._del_eval_video_ffmpeg()
        if trace_file is not None:
            trace_file.close()
        task.close_env()
    return result


def main():
    args = parse_args()
    robotwin_root = Path(args.robotwin_root).resolve()
    if str(robotwin_root) not in sys.path:
        sys.path.insert(0, str(robotwin_root))
    old_cwd = Path.cwd()
    os.chdir(robotwin_root)

    try:
        model, normalizer, metadata = load_birelpose_policy(args.ckpt, device=args.device)
        task_args = prepare_task_args(robotwin_root, args.task_name, args.task_config)
        if args.save_video:
            if args.video_dir is None:
                raise ValueError("--video_dir is required when --save_video is set")
            task_args["data_type"]["rgb"] = True
            task_args["eval_video_save_dir"] = Path(args.video_dir)
            task_args["video_size"] = get_head_camera_video_size(robotwin_root, task_args)
        seeds = load_seed_list(args)
        print_checkpoint_diagnostics(metadata, normalizer)
        print_training_action_distribution(args.training_data_dir or metadata.get("config", {}).get("data_dir"))
        print(f"task_name: {args.task_name}")
        print(f"task_config: {args.task_config}")
        print(f"ckpt: {args.ckpt}")
        print(f"num_episodes: {len(seeds)}")
        print(f"exec_steps: {args.exec_steps}")
        print(f"rollout_mode: {args.rollout_mode}")
        print(f"output_mode: {resolve_output_mode(args, metadata)}")
        print(f"max_steps: {args.max_steps}")
        print(f"expert_check: {args.expert_check}")
        print(f"save_video: {args.save_video}")
        if args.save_video:
            print(f"video_dir: {args.video_dir}")
            print(f"video_size: {task_args['video_size']}")
        print(f"obs_dim: {metadata['obs_dim']}, action_dim: {metadata['action_dim']}")

        task = load_task(robotwin_root, args.task_name)
        results = []
        for episode_id, seed in enumerate(seeds):
            if args.debug_manual_action_test:
                result = run_manual_action_test(task, args.task_name, task_args, seed, episode_id, args)
            else:
                result = rollout_one_seed(
                    task,
                    args.task_name,
                    task_args,
                    model,
                    normalizer,
                    metadata,
                    seed,
                    episode_id,
                    args,
                )
            results.append(result)
            done = len(results)
            success = sum(1 for item in results if item["success"])
            print(
                f"episode={episode_id} seed={seed} success={result['success']} "
                f"steps={result['steps']} expert_ok={result.get('expert_ok')} "
                f"failure_reason={result['failure_reason']} error={result['error']} "
                f"rate={success}/{done}={success / done:.3f}"
            )

        success = sum(1 for item in results if item["success"])
        summary = {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "ckpt": args.ckpt,
            "num_episodes": len(results),
            "success": success,
            "success_rate": success / max(1, len(results)),
            "results": results,
        }
        print("SUMMARY", json.dumps(summary, indent=2))
        if args.output_json is not None:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"wrote: {output_path}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
