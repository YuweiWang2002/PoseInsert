from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

POSEINSERT_ROOT = Path(__file__).resolve().parents[1]
if str(POSEINSERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEINSERT_ROOT))

from robotwin_birelpose.inference import load_birelpose_policy, predict_action_sequence
from robotwin_birelpose.relative_action import relative_action_to_absolute_ee_action


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def quat_angle_batch(q_pred: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    q_pred = q_pred / np.maximum(np.linalg.norm(q_pred, axis=-1, keepdims=True), 1e-12)
    q_gt = q_gt / np.maximum(np.linalg.norm(q_gt, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q_pred * q_gt, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def sample_indices(T: int, n_obs_steps: int, horizon: int, num_samples: int) -> np.ndarray:
    start = n_obs_steps - 1
    end = T - horizon
    if end < start:
        return np.empty((0,), dtype=np.int64)
    candidates = np.arange(start, end + 1)
    if len(candidates) <= num_samples:
        return candidates
    return np.linspace(start, end, num_samples, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--obs_key", default="rel_obs")
    parser.add_argument("--action_key", default="action_ee")
    parser.add_argument("--output_mode", choices=["absolute", "relative"], default="absolute")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    model, normalizer, metadata = load_birelpose_policy(args.ckpt, device=args.device)
    n_obs_steps = metadata["n_obs_steps"]
    horizon = metadata["horizon"]

    paths = sorted(Path(args.data_dir).glob("*.npz"))[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    records = []
    plot_count = 0
    for ep_id, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            rel_obs = np.asarray(data[args.obs_key], dtype=np.float32)
            action = np.asarray(data[args.action_key], dtype=np.float32)
            action_ee = np.asarray(data["action_ee"], dtype=np.float32) if "action_ee" in data.files else None
            object_pose = np.asarray(data["object_pose"], dtype=np.float32) if "object_pose" in data.files else None
        for t in sample_indices(rel_obs.shape[0], n_obs_steps, horizon, args.num_samples):
            obs_window = rel_obs[t - n_obs_steps + 1 : t + 1]
            gt = action[t : t + horizon]
            pred = predict_action_sequence(model, normalizer, obs_window, device=args.device)
            diff = pred - gt
            if args.output_mode == "relative":
                left_xyz_slice = slice(6, 9)
                right_xyz_slice = slice(15, 18)
                left_gripper_idx = 18
                right_gripper_idx = 19
                left_quat_angle_action0 = 0.0
                right_quat_angle_action0 = 0.0
            else:
                left_xyz_slice = slice(0, 3)
                right_xyz_slice = slice(8, 11)
                left_gripper_idx = 7
                right_gripper_idx = 15
                left_quat_angle_action0 = float(
                    quat_angle_batch(pred[None, 0, 3:7], gt[None, 0, 3:7])[0]
                )
                right_quat_angle_action0 = float(
                    quat_angle_batch(pred[None, 0, 11:15], gt[None, 0, 11:15])[0]
                )
            rec = {
                "episode": path.name,
                "t": int(t),
                "action0_l2": float(np.linalg.norm(diff[0])),
                "full_horizon_l2_mean": float(np.linalg.norm(diff, axis=1).mean()),
                "left_xyz_action0_error": float(np.linalg.norm(diff[0, left_xyz_slice])),
                "right_xyz_action0_error": float(np.linalg.norm(diff[0, right_xyz_slice])),
                "left_xyz_full_error_mean": float(np.linalg.norm(diff[:, left_xyz_slice], axis=1).mean()),
                "right_xyz_full_error_mean": float(np.linalg.norm(diff[:, right_xyz_slice], axis=1).mean()),
                "left_gripper_action0_error": float(abs(diff[0, left_gripper_idx])),
                "right_gripper_action0_error": float(abs(diff[0, right_gripper_idx])),
                "left_quat_angle_action0": left_quat_angle_action0,
                "right_quat_angle_action0": right_quat_angle_action0,
                "pred_left_first_to_last_delta": float(
                    np.linalg.norm(pred[-1, left_xyz_slice] - pred[0, left_xyz_slice])
                ),
                "gt_left_first_to_last_delta": float(np.linalg.norm(gt[-1, left_xyz_slice] - gt[0, left_xyz_slice])),
                "pred_right_first_to_last_delta": float(
                    np.linalg.norm(pred[-1, right_xyz_slice] - pred[0, right_xyz_slice])
                ),
                "gt_right_first_to_last_delta": float(
                    np.linalg.norm(gt[-1, right_xyz_slice] - gt[0, right_xyz_slice])
                ),
            }
            if args.output_mode == "relative":
                if action_ee is None or object_pose is None:
                    raise KeyError("relative output_mode requires action_ee and object_pose in each npz")
                obj_seq = object_pose[t : t + horizon]
                gt_abs = action_ee[t : t + horizon]
                pred_abs = relative_action_to_absolute_ee_action(pred, obj_seq)
                abs_diff = pred_abs - gt_abs
                current_obj = np.repeat(object_pose[t : t + 1], horizon, axis=0)
                pred_abs_current_obj = relative_action_to_absolute_ee_action(pred, current_obj)
                abs_diff_current_obj = pred_abs_current_obj - gt_abs
                rec.update(
                    {
                        "abs_left_xyz_action0_error": float(np.linalg.norm(abs_diff[0, :3])),
                        "abs_right_xyz_action0_error": float(np.linalg.norm(abs_diff[0, 8:11])),
                        "abs_left_xyz_full_error_mean": float(np.linalg.norm(abs_diff[:, :3], axis=1).mean()),
                        "abs_right_xyz_full_error_mean": float(np.linalg.norm(abs_diff[:, 8:11], axis=1).mean()),
                        "abs_left_quat_dot_action0": float(
                            np.cos(quat_angle_batch(pred_abs[None, 0, 3:7], gt_abs[None, 0, 3:7])[0] / 2.0)
                        ),
                        "abs_right_quat_dot_action0": float(
                            np.cos(quat_angle_batch(pred_abs[None, 0, 11:15], gt_abs[None, 0, 11:15])[0] / 2.0)
                        ),
                        "abs_left_gripper_action0_error": float(abs(abs_diff[0, 7])),
                        "abs_right_gripper_action0_error": float(abs(abs_diff[0, 15])),
                        "current_object_abs_left_xyz_full_error_mean": float(
                            np.linalg.norm(abs_diff_current_obj[:, :3], axis=1).mean()
                        ),
                        "current_object_abs_right_xyz_full_error_mean": float(
                            np.linalg.norm(abs_diff_current_obj[:, 8:11], axis=1).mean()
                        ),
                    }
                )
            records.append(rec)

            if plot_count < 5:
                if args.output_mode == "relative":
                    plot_specs = [
                        ("left_rel", pred[:, 6:9], gt[:, 6:9], f"pred_vs_gt_left_rel_xyz_ep{ep_id}_t{t}.png"),
                        ("right_rel", pred[:, 15:18], gt[:, 15:18], f"pred_vs_gt_right_rel_xyz_ep{ep_id}_t{t}.png"),
                    ]
                else:
                    plot_specs = [
                        ("left", pred[:, :3], gt[:, :3], f"pred_vs_gt_left_xyz_ep{ep_id}_t{t}.png"),
                        ("right", pred[:, 8:11], gt[:, 8:11], f"pred_vs_gt_right_xyz_ep{ep_id}_t{t}.png"),
                    ]
                for label, pred_xyz, gt_xyz, filename in plot_specs:
                    plt.figure()
                    for i, axis in enumerate(["x", "y", "z"]):
                        plt.plot(pred_xyz[:, i], label=f"pred_{axis}")
                        plt.plot(gt_xyz[:, i], "--", label=f"gt_{axis}")
                    plt.title(f"{label} xyz ep{ep_id} t{t}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(output_dir / filename)
                    plt.close()
                pred_delta = np.linalg.norm(np.diff(plot_specs[0][1], axis=0), axis=1)
                gt_delta = np.linalg.norm(np.diff(plot_specs[0][2], axis=0), axis=1)
                plt.figure()
                plt.plot(pred_delta, label="pred_left_delta")
                plt.plot(gt_delta, "--", label="gt_left_delta")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"pred_vs_gt_delta_norm_ep{ep_id}_t{t}.png")
                plt.close()
                plot_count += 1

    def mean(key):
        return float(np.mean([r[key] for r in records])) if records else 0.0

    summary = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "obs_key": args.obs_key,
        "action_key": args.action_key,
        "output_mode": args.output_mode,
        "num_records": len(records),
        "mean_action0_l2": mean("action0_l2"),
        "mean_full_horizon_l2": mean("full_horizon_l2_mean"),
        "mean_left_xyz_action0_error": mean("left_xyz_action0_error"),
        "mean_right_xyz_action0_error": mean("right_xyz_action0_error"),
        "mean_left_xyz_full_error": mean("left_xyz_full_error_mean"),
        "mean_right_xyz_full_error": mean("right_xyz_full_error_mean"),
        "mean_left_quat_angle_action0": mean("left_quat_angle_action0"),
        "mean_right_quat_angle_action0": mean("right_quat_angle_action0"),
        "mean_pred_left_first_to_last_delta": mean("pred_left_first_to_last_delta"),
        "mean_gt_left_first_to_last_delta": mean("gt_left_first_to_last_delta"),
        "mean_pred_right_first_to_last_delta": mean("pred_right_first_to_last_delta"),
        "mean_gt_right_first_to_last_delta": mean("gt_right_first_to_last_delta"),
        "records": records,
    }
    if args.output_mode == "relative":
        summary.update(
            {
                "mean_abs_left_xyz_action0_error": mean("abs_left_xyz_action0_error"),
                "mean_abs_right_xyz_action0_error": mean("abs_right_xyz_action0_error"),
                "mean_abs_left_xyz_full_error": mean("abs_left_xyz_full_error_mean"),
                "mean_abs_right_xyz_full_error": mean("abs_right_xyz_full_error_mean"),
                "mean_abs_left_quat_dot_action0": mean("abs_left_quat_dot_action0"),
                "mean_abs_right_quat_dot_action0": mean("abs_right_quat_dot_action0"),
                "mean_abs_left_gripper_action0_error": mean("abs_left_gripper_action0_error"),
                "mean_abs_right_gripper_action0_error": mean("abs_right_gripper_action0_error"),
                "mean_current_object_abs_left_xyz_full_error": mean(
                    "current_object_abs_left_xyz_full_error_mean"
                ),
                "mean_current_object_abs_right_xyz_full_error": mean(
                    "current_object_abs_right_xyz_full_error_mean"
                ),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
