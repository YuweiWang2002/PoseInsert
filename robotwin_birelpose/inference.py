from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import BiRelPoseDiffusionPolicy, BiRelPoseMlpPolicy
from .se3_utils import normalize_quat_wxyz


def _normalizer_to_numpy(normalizer: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, value in normalizer.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        out[key] = np.asarray(value, dtype=np.float32)
    return out


def _make_model_from_config(config: dict, obs_dim: int, action_dim: int, horizon: int, n_obs_steps: int):
    model_type = config.get("model_type", "diffusion")
    if model_type == "diffusion":
        return BiRelPoseDiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            obs_feature_dim=int(config.get("obs_feature_dim", 128)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            down_dims=tuple(config.get("down_dims", [128, 256])),
            num_inference_steps=int(config.get("num_inference_steps", 20)),
            prediction=config.get("prediction", "sample"),
        )
    if model_type == "mlp":
        return BiRelPoseMlpPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            hidden_dim=int(config.get("hidden_dim", 256)),
        )
    raise ValueError(f"Unsupported checkpoint model_type {model_type!r}")


def load_birelpose_policy(ckpt_path, device="cpu"):
    """Load a BiRelPose policy checkpoint for inference.

    Returns:
        ``(model, normalizer, metadata)`` where ``model`` is in eval mode and
        ``normalizer`` contains numpy arrays.
    """

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")

    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ["model", "normalizer", "config", "obs_dim", "action_dim", "horizon", "n_obs_steps"]:
        if key not in ckpt:
            raise KeyError(f"checkpoint missing key {key!r}")

    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    horizon = int(ckpt["horizon"])
    n_obs_steps = int(ckpt["n_obs_steps"])
    config = dict(ckpt["config"])

    model = _make_model_from_config(config, obs_dim, action_dim, horizon, n_obs_steps)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    metadata = {
        "ckpt_path": str(ckpt_path),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "config": config,
    }
    return model, _normalizer_to_numpy(ckpt["normalizer"]), metadata


@torch.no_grad()
def predict_action_sequence(model, normalizer: dict[str, np.ndarray], obs_queue, device=None) -> np.ndarray:
    """Predict a denormalized action sequence from an observation queue.

    Args:
        model: Loaded BiRelPose policy.
        normalizer: Dict with ``obs_mean``, ``obs_std``, ``action_mean``, and
            ``action_std``.
        obs_queue: Array-like with shape ``[n_obs_steps, obs_dim]``.

    Returns:
        Denormalized action sequence with shape ``[horizon, action_dim]``.
    """

    obs = np.asarray(obs_queue, dtype=np.float32)
    expected = (model.n_obs_steps, model.obs_dim)
    if obs.shape != expected:
        raise ValueError(f"obs_queue must have shape {expected}, got {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("obs_queue contains NaN or Inf")

    obs_norm = (obs - normalizer["obs_mean"]) / normalizer["obs_std"]
    if device is None:
        device = next(model.parameters()).device
    obs_tensor = torch.as_tensor(obs_norm[None], dtype=torch.float32, device=device)
    action_norm = model.sample_action(obs_tensor).detach().cpu().numpy()[0]
    action = action_norm * normalizer["action_std"] + normalizer["action_mean"]
    if action.shape != (model.horizon, model.action_dim):
        raise ValueError(f"model returned action shape {action.shape}, expected {(model.horizon, model.action_dim)}")
    return action.astype(np.float32, copy=False)


def sanitize_ee_action(action, max_abs_position: float = 5.0) -> np.ndarray:
    """Validate and sanitize one RoboTwin ee action.

    The action format is ``[left_xyz, left_quat_wxyz, left_gripper,
    right_xyz, right_quat_wxyz, right_gripper]`` with shape ``(16,)``.
    Quaternions are normalized in ``wxyz`` order and grippers are clipped to
    ``[0, 1]``. Position values are not clipped; if they exceed
    ``max_abs_position`` meters in absolute value, a clear error is raised.
    """

    out = np.asarray(action, dtype=np.float32).copy()
    if out.shape != (16,):
        raise ValueError(f"ee action must have shape (16,), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("ee action contains NaN or Inf")
    if np.max(np.abs(out[[0, 1, 2, 8, 9, 10]])) > max_abs_position:
        raise ValueError(
            f"ee action position exceeds {max_abs_position}m: "
            f"left={out[:3]}, right={out[8:11]}"
        )

    out[3:7] = normalize_quat_wxyz(out[3:7]).astype(np.float32)
    out[11:15] = normalize_quat_wxyz(out[11:15]).astype(np.float32)
    out[7] = np.clip(out[7], 0.0, 1.0)
    out[15] = np.clip(out[15], 0.0, 1.0)
    return out
