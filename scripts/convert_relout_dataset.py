"""Convert absolute-output BiRelPose npz episodes to relative-output episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotwin_birelpose.relative_action import (
    convert_npz_absolute_to_relative_output,
    relative_action_to_absolute_ee_action,
)


def quat_dot_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return absolute quaternion dot product for wxyz quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.abs(np.sum(a * b, axis=-1))


def round_trip_summary(npz_path: Path) -> dict:
    """Compute reconstruction errors for one converted relative-output npz."""
    with np.load(npz_path, allow_pickle=True) as data:
        action_ee = np.asarray(data["action_ee"], dtype=np.float32)
        rel_action = np.asarray(data["rel_action"], dtype=np.float32)
        object_pose = np.asarray(data["object_pose"], dtype=np.float32)

    recon = relative_action_to_absolute_ee_action(rel_action, object_pose)
    left_xyz_err = np.linalg.norm(recon[:, :3] - action_ee[:, :3], axis=-1)
    right_xyz_err = np.linalg.norm(recon[:, 8:11] - action_ee[:, 8:11], axis=-1)
    left_quat_dot = quat_dot_abs(recon[:, 3:7], action_ee[:, 3:7])
    right_quat_dot = quat_dot_abs(recon[:, 11:15], action_ee[:, 11:15])
    left_gripper_err = np.abs(recon[:, 7] - action_ee[:, 7])
    right_gripper_err = np.abs(recon[:, 15] - action_ee[:, 15])
    return {
        "left_xyz_error_mean": float(left_xyz_err.mean()),
        "left_xyz_error_max": float(left_xyz_err.max()),
        "right_xyz_error_mean": float(right_xyz_err.mean()),
        "right_xyz_error_max": float(right_xyz_err.max()),
        "left_quat_dot_mean": float(left_quat_dot.mean()),
        "left_quat_dot_min": float(left_quat_dot.min()),
        "right_quat_dot_mean": float(right_quat_dot.mean()),
        "right_quat_dot_min": float(right_quat_dot.min()),
        "left_gripper_error_max": float(left_gripper_err.max()),
        "right_gripper_error_max": float(right_gripper_err.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.npz"))
    if args.max_episodes is not None:
        files = files[:args.max_episodes]
    if not files:
        raise FileNotFoundError(f"no npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"num episodes: {len(files)}")

    summaries = []
    for idx, input_npz in enumerate(files):
        output_npz = output_dir / input_npz.name
        if output_npz.exists() and not args.overwrite:
            raise FileExistsError(f"{output_npz} already exists; pass --overwrite to replace")

        convert_npz_absolute_to_relative_output(input_npz, output_npz)
        with np.load(output_npz, allow_pickle=True) as data:
            rel_obs_shape = data["rel_obs"].shape
            action_shape = data["action_ee"].shape
            object_shape = data["object_pose"].shape
            rel_action_shape = data["rel_action"].shape
        summary = round_trip_summary(output_npz)
        summaries.append(summary)
        print(
            f"[{idx}] {input_npz.name}: T={action_shape[0]} "
            f"rel_obs={rel_obs_shape} action_ee={action_shape} "
            f"object_pose={object_shape} rel_action={rel_action_shape}"
        )
        print(
            "    roundtrip "
            f"left_xyz_mean/max={summary['left_xyz_error_mean']:.3e}/{summary['left_xyz_error_max']:.3e} "
            f"right_xyz_mean/max={summary['right_xyz_error_mean']:.3e}/{summary['right_xyz_error_max']:.3e} "
            f"left_quat_dot_mean/min={summary['left_quat_dot_mean']:.9f}/{summary['left_quat_dot_min']:.9f} "
            f"right_quat_dot_mean/min={summary['right_quat_dot_mean']:.9f}/{summary['right_quat_dot_min']:.9f} "
            f"gripper_max={max(summary['left_gripper_error_max'], summary['right_gripper_error_max']):.3e}"
        )

    max_left_xyz = max(item["left_xyz_error_max"] for item in summaries)
    max_right_xyz = max(item["right_xyz_error_max"] for item in summaries)
    min_left_dot = min(item["left_quat_dot_min"] for item in summaries)
    min_right_dot = min(item["right_quat_dot_min"] for item in summaries)
    max_gripper = max(
        max(item["left_gripper_error_max"], item["right_gripper_error_max"]) for item in summaries
    )
    print("aggregate roundtrip:")
    print(f"  left_xyz_error_max: {max_left_xyz:.6e}")
    print(f"  right_xyz_error_max: {max_right_xyz:.6e}")
    print(f"  left_quat_dot_min: {min_left_dot:.9f}")
    print(f"  right_quat_dot_min: {min_right_dot:.9f}")
    print(f"  gripper_error_max: {max_gripper:.6e}")


if __name__ == "__main__":
    main()
