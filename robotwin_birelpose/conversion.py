"""Conversion helpers for RoboTwin endpose demos to BiRelPoseDP arrays.

This first version intentionally requires a real per-frame object pose sequence.
It never fills a full trajectory by repeating a reset-time object pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .hdf5_endpose_reader import RobotWinEndposeEpisode, load_endpose_episode
from .relation_graph import build_bimanual_relation_obs, get_relation_obs_dim


OBJECT_POSE_KEYWORDS = ("object", "actor", "pose", "target", "block")
NON_OBJECT_POSE_PREFIXES = ("/endpose/", "/joint_action/", "/observation/")


@dataclass(frozen=True)
class Hdf5DatasetInfo:
    """Metadata for one HDF5 dataset discovered during inspection."""

    key: str
    shape: tuple[int, ...]
    dtype: str


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("h5py is required for RoboTwin HDF5 conversion") from exc
    return h5py


def find_hdf5_files(input_dir, max_episodes: int | None = None) -> list[Path]:
    """Find RoboTwin episode HDF5 files under ``input_dir``.

    Args:
        input_dir: Directory containing ``episode<N>.hdf5`` files, or a parent
            directory such as a RoboTwin dataset root with a ``data`` subdir.
        max_episodes: Optional maximum number of files to return.

    Returns:
        Sorted list of HDF5 paths.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input_dir does not exist: {root}")
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
    if max_episodes is not None:
        files = files[:max_episodes]
    return files


def list_hdf5_datasets(hdf5_path) -> list[Hdf5DatasetInfo]:
    """List all datasets in an HDF5 file."""
    h5py = _import_h5py()
    path = Path(hdf5_path)
    infos: list[Hdf5DatasetInfo] = []

    with h5py.File(path, "r") as root:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                infos.append(Hdf5DatasetInfo(key="/" + name, shape=tuple(obj.shape), dtype=str(obj.dtype)))

        root.visititems(visit)
    return infos


def find_object_pose_candidates(
    hdf5_path,
    expected_frames: int | None = None,
    keywords: Iterable[str] = OBJECT_POSE_KEYWORDS,
) -> list[Hdf5DatasetInfo]:
    """Find possible object-pose datasets without auto-selecting one.

    Candidate datasets must contain one of ``keywords`` in the path and have a
    final dimension of 7. If ``expected_frames`` is provided, the first
    dimension must match it.
    """
    keyword_tuple = tuple(k.lower() for k in keywords)
    candidates: list[Hdf5DatasetInfo] = []
    for info in list_hdf5_datasets(hdf5_path):
        key_lower = info.key.lower()
        if key_lower.startswith(NON_OBJECT_POSE_PREFIXES):
            continue
        if not any(k in key_lower for k in keyword_tuple):
            continue
        if len(info.shape) < 2 or info.shape[-1] != 7:
            continue
        if expected_frames is not None and info.shape[0] != expected_frames:
            continue
        candidates.append(info)
    return candidates


def load_object_pose_sequence(hdf5_path, object_pose_key: str) -> np.ndarray:
    """Load a per-frame object pose sequence from a specified HDF5 key.

    Args:
        hdf5_path: Episode HDF5 file.
        object_pose_key: Explicit dataset key, e.g. ``/object_pose/main``.

    Returns:
        Array with shape ``(T, 7)`` in ``xyz + qw qx qy qz``.
    """
    h5py = _import_h5py()
    path = Path(hdf5_path)
    key = object_pose_key if object_pose_key.startswith("/") else "/" + object_pose_key
    with h5py.File(path, "r") as root:
        if key not in root:
            raise KeyError(f"{path} does not contain object_pose_key {key!r}")
        arr = np.asarray(root[key][()], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"object pose sequence must have shape (T, 7); got {arr.shape} from {key}")
    return arr


def build_rel_obs_sequence(
    episode: RobotWinEndposeEpisode,
    object_pose: np.ndarray,
    use_left_right_relation: bool = True,
    use_target_relation: bool = False,
    target_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``rel_obs`` for every frame of one episode.

    Args:
        episode: Endpose episode loaded from HDF5.
        object_pose: Per-frame object pose with shape ``(T, 7)``.
        use_left_right_relation: Include ``T_left^-1 T_right``.
        use_target_relation: Include optional target relation.
        target_pose: Optional target pose. If supplied as shape ``(7,)``, it is
            reused for all frames; shape ``(T, 7)`` is also accepted.

    Returns:
        ``float32`` array with shape ``(T, obs_dim)``.
    """
    object_pose = np.asarray(object_pose, dtype=np.float32)
    if object_pose.shape != episode.left_ee_pose.shape:
        raise ValueError(
            f"object_pose must have shape {episode.left_ee_pose.shape}; got {object_pose.shape}"
        )

    if target_pose is not None:
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if target_pose.shape == (7,):
            target_pose = np.repeat(target_pose[None, :], episode.num_frames, axis=0)
        if target_pose.shape != object_pose.shape:
            raise ValueError(f"target_pose must have shape (7,) or {object_pose.shape}; got {target_pose.shape}")

    rel_obs = []
    for i in range(episode.num_frames):
        rel_obs.append(
            build_bimanual_relation_obs(
                left_ee_pose=episode.left_ee_pose[i],
                right_ee_pose=episode.right_ee_pose[i],
                object_pose=object_pose[i],
                left_gripper=float(episode.left_gripper[i]),
                right_gripper=float(episode.right_gripper[i]),
                target_pose=None if target_pose is None else target_pose[i],
                use_left_right_relation=use_left_right_relation,
                use_target_relation=use_target_relation,
            )
        )
    return np.asarray(rel_obs, dtype=np.float32)


def convert_episode_to_birelpose(
    hdf5_path,
    object_pose: np.ndarray,
    task_name: str = "handover_block",
    target_pose: np.ndarray | None = None,
) -> dict:
    """Convert one RoboTwin HDF5 episode to BiRelPoseDP arrays.

    Action convention: ``action_ee`` has shape ``(T, 16)`` and stores the
    per-frame RoboTwin ``ee`` pose/gripper layout from the HDF5 endpose fields.
    A future dataset class may shift this to next-frame actions if desired.
    """
    episode = load_endpose_episode(hdf5_path)
    rel_obs = build_rel_obs_sequence(episode, object_pose=object_pose, target_pose=target_pose)
    expected_dim = get_relation_obs_dim()
    if rel_obs.shape != (episode.num_frames, expected_dim):
        raise ValueError(f"rel_obs must have shape ({episode.num_frames}, {expected_dim}); got {rel_obs.shape}")

    return {
        "rel_obs": rel_obs,
        "action_ee": episode.as_ee_action_array(),
        "left_endpose": episode.left_ee_pose,
        "right_endpose": episode.right_ee_pose,
        "left_gripper": episode.left_gripper[:, None],
        "right_gripper": episode.right_gripper[:, None],
        "object_pose": np.asarray(object_pose, dtype=np.float32),
        "episode_path": str(Path(hdf5_path)),
        "task_name": task_name,
    }


def save_converted_episode(output_path, converted: dict) -> None:
    """Save converted episode arrays to ``.npz``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **converted)
