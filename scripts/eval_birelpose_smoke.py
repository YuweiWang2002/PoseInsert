"""Run a minimal BiRelPoseDP inference smoke test in one RoboTwin task.

This script does not collect data, train, or evaluate success rate. It only
checks that runtime rel_obs -> policy -> ee action -> RoboTwin take_action
forms a working loop for a small number of steps.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import deque
from pathlib import Path
import sys

import numpy as np
import yaml

POSEINSERT_ROOT = Path(__file__).resolve().parents[1]
if str(POSEINSERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEINSERT_ROOT))

from robotwin_birelpose.inference import (
    load_birelpose_policy,
    predict_action_sequence,
    sanitize_ee_action,
)
from robotwin_birelpose.relation_graph import build_bimanual_relation_obs
from robotwin_birelpose.task_adapters import extract_object_poses_from_task


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin_root", required=True)
    parser.add_argument("--task_name", default="handover_block")
    parser.add_argument("--task_config", default="handover_block_pose_debug")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--exec_steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object_index", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_task(robotwin_root: Path, task_name: str):
    sys.path.insert(0, str(robotwin_root))
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _prepare_args(robotwin_root: Path, task_name: str, task_config: str) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    args = _load_yaml(robotwin_root / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_data"] = False
    args["collect_data"] = False
    args["render_freq"] = 0
    args["data_type"]["rgb"] = False
    args["data_type"]["depth"] = False
    args["data_type"]["pointcloud"] = False
    args["data_type"]["endpose"] = True
    args["data_type"]["qpos"] = True

    embodiment_type = args["embodiment"]
    embodiment_types = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

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

    args["left_embodiment_config"] = _load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = _load_yaml(Path(args["right_robot_file"]) / "config.yml")
    return args


def _rel_obs_from_task(task, task_name: str, object_index: int) -> np.ndarray:
    obs = task.get_obs()
    endpose = obs.get("endpose", {})
    required = ["left_endpose", "right_endpose", "left_gripper", "right_gripper"]
    missing = [key for key in required if key not in endpose]
    if missing:
        raise KeyError(f"obs['endpose'] missing keys: {missing}")

    object_result = extract_object_poses_from_task(task, task_name, object_index=object_index)
    rel_obs = build_bimanual_relation_obs(
        left_ee_pose=np.asarray(endpose["left_endpose"], dtype=np.float32),
        right_ee_pose=np.asarray(endpose["right_endpose"], dtype=np.float32),
        object_pose=np.asarray(object_result["object_pose"], dtype=np.float32),
        left_gripper=float(endpose["left_gripper"]),
        right_gripper=float(endpose["right_gripper"]),
    )
    if rel_obs.shape != (29,):
        raise ValueError(f"rel_obs must have shape (29,), got {rel_obs.shape}")
    return rel_obs


def main():
    args = _parse_args()
    robotwin_root = Path(args.robotwin_root).resolve()
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"robotwin_root does not exist: {robotwin_root}")
    if args.exec_steps <= 0:
        raise ValueError("exec_steps must be positive")

    model, normalizer, metadata = load_birelpose_policy(args.ckpt, device=args.device)
    print(f"ckpt: {args.ckpt}")
    print(f"obs_dim: {metadata['obs_dim']}")
    print(f"action_dim: {metadata['action_dim']}")
    print(f"horizon: {metadata['horizon']}")
    print(f"n_obs_steps: {metadata['n_obs_steps']}")
    print(f"normalizer_loaded: {bool(normalizer)}")

    old_cwd = Path.cwd()
    os.chdir(robotwin_root)
    completed_steps = 0
    task = None
    try:
        for episode_id in range(args.num_episodes):
            task = _load_task(robotwin_root, args.task_name)
            setup_args = _prepare_args(robotwin_root, args.task_name, args.task_config)
            task.setup_demo(now_ep_num=episode_id, seed=args.seed + episode_id, **setup_args)
            print(f"episode {episode_id}: reset ok")

            first_rel_obs = _rel_obs_from_task(task, args.task_name, args.object_index)
            obs_queue = deque([first_rel_obs.copy() for _ in range(metadata["n_obs_steps"])], maxlen=metadata["n_obs_steps"])
            print(f"episode {episode_id}: initial rel_obs shape {first_rel_obs.shape}")

            step = 0
            while step < args.max_steps:
                obs_arr = np.stack(list(obs_queue), axis=0)
                action_seq = predict_action_sequence(model, normalizer, obs_arr, device=args.device)
                print(f"episode {episode_id} step {step}: action_seq shape {action_seq.shape}")

                for local_i in range(min(args.exec_steps, action_seq.shape[0])):
                    action = sanitize_ee_action(action_seq[local_i])
                    print(
                        f"episode {episode_id} step {step}: "
                        f"action shape {action.shape}, left_xyz={action[:3]}, right_xyz={action[8:11]}"
                    )
                    task.take_action(action, action_type="ee")
                    completed_steps += 1
                    step += 1
                    if step >= args.max_steps:
                        break

                    rel_obs = _rel_obs_from_task(task, args.task_name, args.object_index)
                    obs_queue.append(rel_obs)

            print(f"episode {episode_id}: completed {step} control steps")
            task.close_env(clear_cache=False)
            task = None
    finally:
        if task is not None:
            task.close_env(clear_cache=False)
        os.chdir(old_cwd)

    print(f"smoke_test_done: completed_steps={completed_steps}")


if __name__ == "__main__":
    main()
