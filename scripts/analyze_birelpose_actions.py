from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def delta_stats(xyz: np.ndarray) -> dict:
    deltas = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return {
        "mean": float(deltas.mean()) if deltas.size else 0.0,
        "max": float(deltas.max()) if deltas.size else 0.0,
        "values": deltas.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.npz"))[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    plt = _setup_matplotlib()
    summaries = []
    all_actions = []
    first_episode = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            rel_obs = np.asarray(data["rel_obs"], dtype=np.float32)
            action = np.asarray(data["action_ee"], dtype=np.float32)
        left_xyz = action[:, :3]
        right_xyz = action[:, 8:11]
        left_delta = delta_stats(left_xyz)
        right_delta = delta_stats(right_xyz)
        quat_left = np.linalg.norm(action[:, 3:7], axis=1)
        quat_right = np.linalg.norm(action[:, 11:15], axis=1)
        summaries.append(
            {
                "path": str(path),
                "rel_obs_shape": list(rel_obs.shape),
                "action_ee_shape": list(action.shape),
                "left_xyz": stats(left_xyz),
                "right_xyz": stats(right_xyz),
                "left_xyz_step_delta_mean": left_delta["mean"],
                "left_xyz_step_delta_max": left_delta["max"],
                "right_xyz_step_delta_mean": right_delta["mean"],
                "right_xyz_step_delta_max": right_delta["max"],
                "left_gripper_range": [float(action[:, 7].min()), float(action[:, 7].max())],
                "right_gripper_range": [float(action[:, 15].min()), float(action[:, 15].max())],
                "left_quat_norm_range": [float(quat_left.min()), float(quat_left.max())],
                "right_quat_norm_range": [float(quat_right.min()), float(quat_right.max())],
            }
        )
        all_actions.append(action)
        if first_episode is None:
            first_episode = (path.stem, action, left_delta["values"], right_delta["values"])

    all_action = np.concatenate(all_actions, axis=0)
    all_left_delta = np.linalg.norm(np.diff(all_action[:, :3], axis=0), axis=1)
    all_right_delta = np.linalg.norm(np.diff(all_action[:, 8:11], axis=0), axis=1)
    summary = {
        "data_dir": str(data_dir),
        "episode_count": len(paths),
        "episodes": summaries,
        "aggregate": {
            "action_shape": list(all_action.shape),
            "left_xyz_step_delta_mean": float(all_left_delta.mean()),
            "left_xyz_step_delta_max": float(all_left_delta.max()),
            "right_xyz_step_delta_mean": float(all_right_delta.mean()),
            "right_xyz_step_delta_max": float(all_right_delta.max()),
            "left_gripper_range": [float(all_action[:, 7].min()), float(all_action[:, 7].max())],
            "right_gripper_range": [float(all_action[:, 15].min()), float(all_action[:, 15].max())],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    name, action, left_delta_values, right_delta_values = first_episode
    t = np.arange(action.shape[0])
    for title, cols, filename in [
        ("left_xyz_vs_t", action[:, :3], "left_xyz_vs_t.png"),
        ("right_xyz_vs_t", action[:, 8:11], "right_xyz_vs_t.png"),
    ]:
        plt.figure()
        for i, axis in enumerate(["x", "y", "z"]):
            plt.plot(t, cols[:, i], label=axis)
        plt.title(f"{title} {name}")
        plt.xlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    plt.figure()
    plt.plot(left_delta_values, label="left")
    plt.plot(right_delta_values, label="right")
    plt.title(f"delta_norm_vs_t {name}")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "delta_norm_vs_t.png")
    plt.close()

    plt.figure()
    plt.plot(t, action[:, 7], label="left")
    plt.plot(t, action[:, 15], label="right")
    plt.title(f"gripper_vs_t {name}")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gripper_vs_t.png")
    plt.close()

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
