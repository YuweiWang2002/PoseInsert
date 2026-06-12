"""Object-relative action representation for RoboTwin bimanual EE actions.

All poses use RoboTwin's confirmed convention: ``xyz + qw qx qy qz``.
Absolute ``ee`` actions use:
``[left_xyz, left_quat_wxyz, left_gripper, right_xyz, right_quat_wxyz, right_gripper]``.
Relative actions use:
``[phi(T_obj^-1 T_left), phi(T_obj^-1 T_right), left_gripper, right_gripper]``,
where ``phi(T) = [R[:, 0], R[:, 1], xyz]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .se3_utils import (
    compose_transform,
    mat_to_pose7_wxyz,
    mat_to_rot6d_xyz,
    pose7_wxyz_to_mat,
    relative_transform,
)


_EPS = 1e-8


def _as_float64_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _check_last_dim(arr: np.ndarray, dim: int, name: str) -> None:
    if arr.shape[-1:] != (dim,):
        raise ValueError(f"{name} must have shape (..., {dim}); got {arr.shape}")


def _check_matching_batch(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(
            f"{a_name} and {b_name} must have matching leading dims; "
            f"got {a.shape} and {b.shape}"
        )


def _rot6d_xyz_to_mat(rot6d_xyz) -> np.ndarray:
    """Convert ``[R[:,0], R[:,1], xyz]`` to homogeneous transform(s).

    The first two rotation columns are orthonormalized using the standard 6D
    rotation representation recovery rule:
    normalize x, remove x component from y, normalize y, then z = x cross y.
    """
    arr = _as_float64_array(rot6d_xyz, "rot6d_xyz")
    _check_last_dim(arr, 9, "rot6d_xyz")

    x = arr[..., 0:3]
    y = arr[..., 3:6]
    xyz = arr[..., 6:9]

    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(x_norm < _EPS):
        raise ValueError("rot6d_xyz contains a near-zero first rotation column")
    x = x / x_norm

    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(y_norm < _EPS):
        raise ValueError("rot6d_xyz contains degenerate rotation columns")
    y = y / y_norm

    z = np.cross(x, y, axis=-1)

    T = np.zeros(arr.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, 0] = x
    T[..., :3, 1] = y
    T[..., :3, 2] = z
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def _action_ee_to_pose7(action_ee: np.ndarray, start: int) -> np.ndarray:
    xyz = action_ee[..., start:start + 3]
    quat = action_ee[..., start + 3:start + 7]
    return np.concatenate([xyz, quat], axis=-1)


def absolute_ee_action_to_relative_action(action_ee, object_pose) -> np.ndarray:
    """Convert world-frame RoboTwin ``ee`` action(s) to object-relative action(s).

    Args:
        action_ee: Array with shape ``(16,)`` or ``(..., 16)`` in
            ``[left_xyz, left_quat_wxyz, left_gripper, right_xyz,
            right_quat_wxyz, right_gripper]`` format.
        object_pose: Array with shape ``(7,)`` or ``(..., 7)`` in
            ``xyz + qw qx qy qz`` format. Leading dimensions must match
            ``action_ee``.

    Returns:
        ``float32`` array with shape ``(20,)`` or ``(..., 20)``:
        ``[phi(T_obj^-1 T_left), phi(T_obj^-1 T_right), left_gripper,
        right_gripper]``. Gripper values are preserved as stored in
        ``action_ee``; no clipping is applied in this direction.
    """
    action_arr = _as_float64_array(action_ee, "action_ee")
    object_arr = _as_float64_array(object_pose, "object_pose")
    _check_last_dim(action_arr, 16, "action_ee")
    _check_last_dim(object_arr, 7, "object_pose")
    _check_matching_batch(action_arr, object_arr, "action_ee", "object_pose")

    T_obj = pose7_wxyz_to_mat(object_arr)
    T_left = pose7_wxyz_to_mat(_action_ee_to_pose7(action_arr, 0))
    T_right = pose7_wxyz_to_mat(_action_ee_to_pose7(action_arr, 8))

    rel_left = mat_to_rot6d_xyz(relative_transform(T_obj, T_left))
    rel_right = mat_to_rot6d_xyz(relative_transform(T_obj, T_right))
    grippers = np.stack([action_arr[..., 7], action_arr[..., 15]], axis=-1)
    return np.concatenate([rel_left, rel_right, grippers], axis=-1).astype(np.float32)


def relative_action_to_absolute_ee_action(rel_action, object_pose) -> np.ndarray:
    """Convert object-relative action(s) back to executable RoboTwin ``ee`` action(s).

    Args:
        rel_action: Array with shape ``(20,)`` or ``(..., 20)`` in
            ``[phi(T_obj^-1 T_left), phi(T_obj^-1 T_right), left_gripper,
            right_gripper]`` format.
        object_pose: Array with shape ``(7,)`` or ``(..., 7)`` in
            ``xyz + qw qx qy qz`` format. Leading dimensions must match
            ``rel_action``.

    Returns:
        ``float32`` array with shape ``(16,)`` or ``(..., 16)`` in RoboTwin
        world-frame absolute ``ee`` action format. Quaternion order is wxyz.
        Gripper values are clipped to ``[0, 1]`` for safe execution.
    """
    rel_arr = _as_float64_array(rel_action, "rel_action")
    object_arr = _as_float64_array(object_pose, "object_pose")
    _check_last_dim(rel_arr, 20, "rel_action")
    _check_last_dim(object_arr, 7, "object_pose")
    _check_matching_batch(rel_arr, object_arr, "rel_action", "object_pose")

    T_obj = pose7_wxyz_to_mat(object_arr)
    T_obj_left = _rot6d_xyz_to_mat(rel_arr[..., :9])
    T_obj_right = _rot6d_xyz_to_mat(rel_arr[..., 9:18])

    T_left = compose_transform(T_obj, T_obj_left)
    T_right = compose_transform(T_obj, T_obj_right)
    left_pose = mat_to_pose7_wxyz(T_left)
    right_pose = mat_to_pose7_wxyz(T_right)
    left_gripper = np.clip(rel_arr[..., 18:19], 0.0, 1.0)
    right_gripper = np.clip(rel_arr[..., 19:20], 0.0, 1.0)

    action = np.concatenate(
        [
            left_pose[..., :7],
            left_gripper,
            right_pose[..., :7],
            right_gripper,
        ],
        axis=-1,
    )
    return action.astype(np.float32)


def convert_npz_absolute_to_relative_output(input_npz, output_npz) -> None:
    """Convert one absolute-output BiRelPose episode npz to relative-output npz.

    The input file must contain ``rel_obs``, ``action_ee``, and ``object_pose``.
    The output keeps existing fields, adds ``rel_action`` with shape ``(T, 20)``,
    and preserves ``action_ee`` for debugging and round-trip checks.
    """
    input_path = Path(input_npz)
    output_path = Path(output_npz)
    with np.load(input_path, allow_pickle=True) as data:
        required = ("rel_obs", "action_ee", "object_pose")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{input_path} is missing required keys: {missing}")

        action_ee = np.asarray(data["action_ee"], dtype=np.float32)
        object_pose = np.asarray(data["object_pose"], dtype=np.float32)
        if action_ee.ndim != 2 or action_ee.shape[1] != 16:
            raise ValueError(f"action_ee must have shape (T, 16); got {action_ee.shape}")
        if object_pose.ndim != 2 or object_pose.shape[1] != 7:
            raise ValueError(f"object_pose must have shape (T, 7); got {object_pose.shape}")
        if action_ee.shape[0] != object_pose.shape[0]:
            raise ValueError(
                f"action_ee and object_pose must have the same T; got "
                f"{action_ee.shape[0]} and {object_pose.shape[0]}"
            )

        converted = {key: data[key] for key in data.files}
        converted["rel_action"] = absolute_ee_action_to_relative_action(action_ee, object_pose)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **converted)
