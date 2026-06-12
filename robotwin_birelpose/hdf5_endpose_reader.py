"""Reader for RoboTwin HDF5 demonstrations with saved end-effector poses.

This module only reads fields that RoboTwin saves when ``data_type.endpose`` is
enabled. It intentionally does not assume object or target poses exist in HDF5;
those should come from a later task adapter or replay extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ENDPOSE_DATASETS = {
    "left_ee_pose": "/endpose/left_endpose",
    "right_ee_pose": "/endpose/right_endpose",
    "left_gripper": "/endpose/left_gripper",
    "right_gripper": "/endpose/right_gripper",
}


@dataclass(frozen=True)
class RobotWinEndposeEpisode:
    """A single RoboTwin episode containing endpose-only robot state.

    Attributes:
        path: Source HDF5 path.
        left_ee_pose: Array with shape ``(T, 7)`` in ``xyz + qw qx qy qz``.
        right_ee_pose: Array with shape ``(T, 7)`` in ``xyz + qw qx qy qz``.
        left_gripper: Array with shape ``(T,)``; normalized RoboTwin gripper
            state in ``[0, 1]``.
        right_gripper: Array with shape ``(T,)``; normalized RoboTwin gripper
            state in ``[0, 1]``.
    """

    path: Path
    left_ee_pose: np.ndarray
    right_ee_pose: np.ndarray
    left_gripper: np.ndarray
    right_gripper: np.ndarray

    def __post_init__(self) -> None:
        _validate_episode_arrays(
            self.left_ee_pose,
            self.right_ee_pose,
            self.left_gripper,
            self.right_gripper,
        )

    @property
    def num_frames(self) -> int:
        """Return the number of frames in this episode."""
        return int(self.left_ee_pose.shape[0])

    def as_ee_action_array(self) -> np.ndarray:
        """Return frames in RoboTwin ``ee`` action layout.

        The returned shape is ``(T, 16)`` with layout:
        ``[left_xyz, left_quat_wxyz, left_gripper,
        right_xyz, right_quat_wxyz, right_gripper]``.

        This is a per-frame pose array, not a shifted policy target. A dataset
        adapter can later choose whether to use the current frame, next frame,
        or a horizon of future frames as actions.
        """
        return make_ee_action_array(
            self.left_ee_pose,
            self.right_ee_pose,
            self.left_gripper,
            self.right_gripper,
        )


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("h5py is required to read RoboTwin HDF5 demonstrations") from exc
    return h5py


def _read_dataset(root: Any, dataset_path: str, hdf5_path: Path) -> np.ndarray:
    if dataset_path not in root:
        raise KeyError(f"{hdf5_path} is missing required dataset {dataset_path!r}")
    return np.asarray(root[dataset_path][()])


def _validate_episode_arrays(
    left_ee_pose: np.ndarray,
    right_ee_pose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
) -> None:
    if left_ee_pose.ndim != 2 or left_ee_pose.shape[1] != 7:
        raise ValueError(f"left_ee_pose must have shape (T, 7); got {left_ee_pose.shape}")
    if right_ee_pose.ndim != 2 or right_ee_pose.shape[1] != 7:
        raise ValueError(f"right_ee_pose must have shape (T, 7); got {right_ee_pose.shape}")
    if left_gripper.ndim != 1:
        raise ValueError(f"left_gripper must have shape (T,); got {left_gripper.shape}")
    if right_gripper.ndim != 1:
        raise ValueError(f"right_gripper must have shape (T,); got {right_gripper.shape}")

    lengths = {
        left_ee_pose.shape[0],
        right_ee_pose.shape[0],
        left_gripper.shape[0],
        right_gripper.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError(
            "endpose arrays must have matching frame count; got "
            f"left_ee={left_ee_pose.shape[0]}, right_ee={right_ee_pose.shape[0]}, "
            f"left_gripper={left_gripper.shape[0]}, right_gripper={right_gripper.shape[0]}"
        )


def load_endpose_episode(hdf5_path) -> RobotWinEndposeEpisode:
    """Load left/right end-effector poses and grippers from a RoboTwin HDF5.

    Args:
        hdf5_path: Path to one ``episode<N>.hdf5`` file.

    Returns:
        ``RobotWinEndposeEpisode`` with arrays converted to ``float32``:
        ``left_ee_pose`` and ``right_ee_pose`` have shape ``(T, 7)`` using
        ``xyz + qw qx qy qz``; grippers have shape ``(T,)``.

    Raises:
        FileNotFoundError: If ``hdf5_path`` does not exist.
        KeyError: If any required ``/endpose/*`` dataset is missing.
        ValueError: If dataset shapes are not consistent.
    """
    path = Path(hdf5_path)
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin HDF5 file does not exist: {path}")

    h5py = _import_h5py()
    with h5py.File(path, "r") as root:
        left_ee_pose = _read_dataset(root, ENDPOSE_DATASETS["left_ee_pose"], path).astype(np.float32)
        right_ee_pose = _read_dataset(root, ENDPOSE_DATASETS["right_ee_pose"], path).astype(np.float32)
        left_gripper = _read_dataset(root, ENDPOSE_DATASETS["left_gripper"], path).astype(np.float32)
        right_gripper = _read_dataset(root, ENDPOSE_DATASETS["right_gripper"], path).astype(np.float32)

    return RobotWinEndposeEpisode(
        path=path,
        left_ee_pose=left_ee_pose,
        right_ee_pose=right_ee_pose,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )


def make_ee_action_array(
    left_ee_pose,
    right_ee_pose,
    left_gripper,
    right_gripper,
) -> np.ndarray:
    """Pack endpose arrays into RoboTwin ``ee`` action layout.

    Args:
        left_ee_pose: Shape ``(T, 7)``, ``xyz + qw qx qy qz``.
        right_ee_pose: Shape ``(T, 7)``, ``xyz + qw qx qy qz``.
        left_gripper: Shape ``(T,)``, normalized gripper in ``[0, 1]``.
        right_gripper: Shape ``(T,)``, normalized gripper in ``[0, 1]``.

    Returns:
        Array with shape ``(T, 16)`` and layout:
        ``[left_xyz, left_quat_wxyz, left_gripper,
        right_xyz, right_quat_wxyz, right_gripper]``.
    """
    left_ee_pose = np.asarray(left_ee_pose, dtype=np.float32)
    right_ee_pose = np.asarray(right_ee_pose, dtype=np.float32)
    left_gripper = np.asarray(left_gripper, dtype=np.float32)
    right_gripper = np.asarray(right_gripper, dtype=np.float32)
    _validate_episode_arrays(left_ee_pose, right_ee_pose, left_gripper, right_gripper)

    return np.concatenate(
        [
            left_ee_pose,
            left_gripper[:, None],
            right_ee_pose,
            right_gripper[:, None],
        ],
        axis=-1,
    )
