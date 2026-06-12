"""Inspect RoboTwin runtime task poses through robotwin_birelpose adapters.

This script resets one RoboTwin task, prints end-effector observation fields,
and prints object/target poses extracted from the live task instance. It does
not train, collect data, or write demonstrations.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import yaml

POSEINSERT_ROOT = Path(__file__).resolve().parents[1]
if str(POSEINSERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEINSERT_ROOT))

from robotwin_birelpose.task_adapters import (
    extract_object_poses_from_task,
    extract_target_pose_from_task,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin_root", required=True, help="Path to RoboTwin repository root")
    parser.add_argument("--task_name", required=True, help="RoboTwin task name, e.g. place_a2b_right")
    parser.add_argument("--task_config", default="demo_clean", help="RoboTwin task config name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object_index", type=int, default=0)
    parser.add_argument("--target_index", type=int, default=0)
    return parser.parse_args()


def _load_task(robotwin_root: Path, task_name: str):
    sys.path.insert(0, str(robotwin_root))
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


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
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    embodiment_types = _load_yaml(embodiment_config_path)

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

    def get_embodiment_config(robot_file):
        return _load_yaml(Path(robot_file) / "config.yml")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def main():
    cli_args = _parse_args()
    robotwin_root = Path(cli_args.robotwin_root).resolve()
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"robotwin_root does not exist: {robotwin_root}")

    old_cwd = Path.cwd()
    os.chdir(robotwin_root)
    task = None
    try:
        task = _load_task(robotwin_root, cli_args.task_name)
        args = _prepare_args(robotwin_root, cli_args.task_name, cli_args.task_config)
        task.setup_demo(now_ep_num=0, seed=cli_args.seed, **args)
        obs = task.get_obs()

        print("obs root keys:", list(obs.keys()))
        print("endpose keys:", list(obs.get("endpose", {}).keys()))
        print("left_endpose:", obs["endpose"].get("left_endpose"))
        print("right_endpose:", obs["endpose"].get("right_endpose"))
        print("left_gripper:", obs["endpose"].get("left_gripper"))
        print("right_gripper:", obs["endpose"].get("right_gripper"))

        object_result = extract_object_poses_from_task(
            task,
            cli_args.task_name,
            object_index=cli_args.object_index,
        )
        print("object_name:", object_result["object_name"])
        print("object_pose:", object_result["object_pose"])
        print("all_object_names:", list(object_result.get("all_object_poses", {}).keys()))

        target_result = extract_target_pose_from_task(
            task,
            cli_args.task_name,
            target_index=cli_args.target_index,
        )
        if target_result is None:
            print("target_name: None")
            print("target_pose: None")
        else:
            print("target_name:", target_result["target_name"])
            print("target_pose:", target_result["target_pose"])
    finally:
        if task is not None:
            task.close_env(clear_cache=False)
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
