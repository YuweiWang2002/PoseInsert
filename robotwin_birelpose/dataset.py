from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BiRelPoseEpisode:
    """One converted RoboTwin episode loaded from an npz file."""

    path: Path
    rel_obs: np.ndarray
    action_ee: np.ndarray


class BiRelPoseNpzDataset:
    """Windowed dataset for converted BiRelPoseDP npz episodes.

    Each sample uses an observation window
    ``[t - n_obs_steps + 1, ..., t]`` and an action window
    ``[t, ..., t + horizon - 1]``. This matches the current converted
    ``action_ee`` convention where each frame stores the executable ee action
    label for that frame.
    """

    def __init__(
        self,
        data_dir,
        n_obs_steps: int = 2,
        horizon: int = 16,
        obs_key: str = "rel_obs",
        action_key: str = "action_ee",
        obs_dim: int = 29,
        action_dim: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.obs_key = obs_key
        self.action_key = action_key
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        if self.n_obs_steps <= 0:
            raise ValueError(f"n_obs_steps must be positive, got {self.n_obs_steps}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        self.episodes = self._load_episodes(sorted(self.data_dir.glob("*.npz")))
        if not self.episodes:
            raise ValueError(f"No .npz episodes found in {self.data_dir}")

        self.index: list[tuple[int, int]] = []
        min_t = self.n_obs_steps - 1
        for episode_id, episode in enumerate(self.episodes):
            max_t = episode.rel_obs.shape[0] - self.horizon
            for t in range(min_t, max_t + 1):
                self.index.append((episode_id, t))
        if not self.index:
            raise ValueError(
                "No valid windows. Need T >= n_obs_steps + horizon - 1 "
                f"for at least one episode; got n_obs_steps={self.n_obs_steps}, "
                f"horizon={self.horizon}."
            )

    def _load_episodes(self, paths: Iterable[Path]) -> list[BiRelPoseEpisode]:
        episodes: list[BiRelPoseEpisode] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if self.obs_key not in data:
                    raise KeyError(f"{path} missing key {self.obs_key!r}")
                if self.action_key not in data:
                    raise KeyError(f"{path} missing key {self.action_key!r}")
                rel_obs = np.asarray(data[self.obs_key], dtype=np.float32)
                action_ee = np.asarray(data[self.action_key], dtype=np.float32)

            if rel_obs.ndim != 2 or rel_obs.shape[1] != self.obs_dim:
                raise ValueError(
                    f"{path}:{self.obs_key} must have shape [T, {self.obs_dim}], "
                    f"got {rel_obs.shape}"
                )
            if action_ee.ndim != 2 or action_ee.shape[1] != self.action_dim:
                raise ValueError(
                    f"{path}:{self.action_key} must have shape [T, {self.action_dim}], "
                    f"got {action_ee.shape}"
                )
            if rel_obs.shape[0] != action_ee.shape[0]:
                raise ValueError(
                    f"{path} has mismatched T: {self.obs_key}={rel_obs.shape[0]}, "
                    f"{self.action_key}={action_ee.shape[0]}"
                )
            if rel_obs.shape[0] < self.n_obs_steps + self.horizon - 1:
                continue
            episodes.append(BiRelPoseEpisode(path=path, rel_obs=rel_obs, action_ee=action_ee))
        return episodes

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        episode_id, t = self.index[idx]
        episode = self.episodes[episode_id]
        obs_start = t - self.n_obs_steps + 1
        action_end = t + self.horizon
        return {
            "obs": episode.rel_obs[obs_start : t + 1].astype(np.float32, copy=False),
            "action": episode.action_ee[t:action_end].astype(np.float32, copy=False),
            "episode_id": episode_id,
            "t": t,
        }


def compute_normalizer(dataset: BiRelPoseNpzDataset, eps: float = 1e-6) -> dict[str, np.ndarray]:
    """Compute mean/std for all rel_obs frames and action_ee frames in a dataset."""

    obs = np.concatenate([episode.rel_obs for episode in dataset.episodes], axis=0)
    action = np.concatenate([episode.action_ee for episode in dataset.episodes], axis=0)
    return {
        "obs_mean": obs.mean(axis=0).astype(np.float32),
        "obs_std": (obs.std(axis=0) + eps).astype(np.float32),
        "action_mean": action.mean(axis=0).astype(np.float32),
        "action_std": (action.std(axis=0) + eps).astype(np.float32),
    }


def save_normalizer(path, normalizer: dict[str, np.ndarray]) -> None:
    """Save a normalizer dict to an npz file."""

    np.savez(path, **normalizer)


def load_normalizer(path) -> dict[str, np.ndarray]:
    """Load a normalizer dict saved by :func:`save_normalizer`."""

    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}


def normalize_batch(batch: dict, normalizer: dict[str, np.ndarray], to_torch: bool = True):
    """Normalize a dataloader batch.

    ``batch`` must contain ``obs`` with shape ``[B, n_obs_steps, 29]`` and
    ``action`` with shape ``[B, horizon, 16]``. If ``to_torch`` is true, the
    returned arrays are converted to ``torch.float32`` tensors.
    """

    obs = (np.asarray(batch["obs"], dtype=np.float32) - normalizer["obs_mean"]) / normalizer["obs_std"]
    action = (
        np.asarray(batch["action"], dtype=np.float32) - normalizer["action_mean"]
    ) / normalizer["action_std"]
    if not to_torch:
        return {"obs": obs.astype(np.float32), "action": action.astype(np.float32)}

    import torch

    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "action": torch.as_tensor(action, dtype=torch.float32),
    }
