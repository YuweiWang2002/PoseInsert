"""Small SE(3) utilities for RoboTwin pose-only relative pose features.

All pose7 values in this module use RoboTwin's confirmed convention:
``xyz + qw qx qy qz``. The quaternion order is wxyz, not xyzw.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _as_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _check_last_dim(arr: np.ndarray, dim: int, name: str) -> None:
    if arr.shape[-1:] != (dim,):
        raise ValueError(f"{name} must have shape (..., {dim}); got {arr.shape}")


def _check_transform_shape(arr: np.ndarray, name: str) -> None:
    if arr.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape (..., 4, 4); got {arr.shape}")


def normalize_quat_wxyz(q) -> np.ndarray:
    """Normalize quaternion(s) in ``qw qx qy qz`` order.

    Args:
        q: Array-like with shape ``(4,)`` or ``(..., 4)``.

    Returns:
        A numpy array with the same shape as ``q`` and unit-norm quaternions.

    Raises:
        ValueError: If the final dimension is not 4 or a quaternion has near
            zero norm.
    """
    q_arr = _as_array(q, "q")
    _check_last_dim(q_arr, 4, "q")
    norm = np.linalg.norm(q_arr, axis=-1, keepdims=True)
    if np.any(norm < _EPS):
        raise ValueError("q contains a zero-norm quaternion")
    return q_arr / norm


def quat_wxyz_to_mat(q) -> np.ndarray:
    """Convert quaternion(s) in ``qw qx qy qz`` order to rotation matrices.

    Args:
        q: Array-like with shape ``(4,)`` or ``(..., 4)``.

    Returns:
        Rotation matrix array with shape ``(3, 3)`` or ``(..., 3, 3)``.
    """
    qn = normalize_quat_wxyz(q)
    w, x, y, z = np.moveaxis(qn, -1, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.empty(qn.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def mat_to_quat_wxyz(R) -> np.ndarray:
    """Convert rotation matrix/matrices to ``qw qx qy qz`` quaternions.

    Args:
        R: Array-like with shape ``(3, 3)`` or ``(..., 3, 3)``.

    Returns:
        Quaternion array with shape ``(4,)`` or ``(..., 4)``.
    """
    R_arr = _as_array(R, "R")
    if R_arr.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3); got {R_arr.shape}")

    flat_R = R_arr.reshape((-1, 3, 3))
    flat_q = np.empty((flat_R.shape[0], 4), dtype=np.float64)

    for i, mat in enumerate(flat_R):
        trace = np.trace(mat)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            flat_q[i, 0] = 0.25 * s
            flat_q[i, 1] = (mat[2, 1] - mat[1, 2]) / s
            flat_q[i, 2] = (mat[0, 2] - mat[2, 0]) / s
            flat_q[i, 3] = (mat[1, 0] - mat[0, 1]) / s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            flat_q[i, 0] = (mat[2, 1] - mat[1, 2]) / s
            flat_q[i, 1] = 0.25 * s
            flat_q[i, 2] = (mat[0, 1] + mat[1, 0]) / s
            flat_q[i, 3] = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            flat_q[i, 0] = (mat[0, 2] - mat[2, 0]) / s
            flat_q[i, 1] = (mat[0, 1] + mat[1, 0]) / s
            flat_q[i, 2] = 0.25 * s
            flat_q[i, 3] = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            flat_q[i, 0] = (mat[1, 0] - mat[0, 1]) / s
            flat_q[i, 1] = (mat[0, 2] + mat[2, 0]) / s
            flat_q[i, 2] = (mat[1, 2] + mat[2, 1]) / s
            flat_q[i, 3] = 0.25 * s

    q = flat_q.reshape(R_arr.shape[:-2] + (4,))
    q = normalize_quat_wxyz(q)
    if q.shape[-1] == 4:
        q = np.where(q[..., :1] < 0.0, -q, q)
    return q


def pose7_wxyz_to_mat(pose) -> np.ndarray:
    """Convert ``xyz + qw qx qy qz`` pose(s) to homogeneous matrices.

    Args:
        pose: Array-like with shape ``(7,)`` or ``(..., 7)``.

    Returns:
        Homogeneous transform(s) with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Raises:
        ValueError: If ``pose`` does not end with dimension 7 or contains a
            zero-norm quaternion.
    """
    pose_arr = _as_array(pose, "pose")
    _check_last_dim(pose_arr, 7, "pose")

    xyz = pose_arr[..., :3]
    quat = pose_arr[..., 3:]
    R = quat_wxyz_to_mat(quat)

    T = np.zeros(pose_arr.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def mat_to_pose7_wxyz(T) -> np.ndarray:
    """Convert homogeneous matrix/matrices to ``xyz + qw qx qy qz`` pose(s).

    Args:
        T: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Returns:
        Pose array with shape ``(7,)`` or ``(..., 7)``.
    """
    T_arr = _as_array(T, "T")
    _check_transform_shape(T_arr, "T")
    xyz = T_arr[..., :3, 3]
    quat = mat_to_quat_wxyz(T_arr[..., :3, :3])
    return np.concatenate([xyz, quat], axis=-1)


def invert_transform(T) -> np.ndarray:
    """Return the inverse of homogeneous transform(s).

    Args:
        T: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Returns:
        Inverse transform(s) with the same shape as ``T``.
    """
    T_arr = _as_array(T, "T")
    _check_transform_shape(T_arr, "T")
    R = T_arr[..., :3, :3]
    t = T_arr[..., :3, 3]
    R_inv = np.swapaxes(R, -1, -2)

    T_inv = np.zeros_like(T_arr, dtype=np.float64)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = -(R_inv @ t[..., None])[..., 0]
    T_inv[..., 3, 3] = 1.0
    return T_inv


def compose_transform(A, B) -> np.ndarray:
    """Compose homogeneous transforms as ``A @ B``.

    Args:
        A: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.
        B: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Returns:
        Matrix product ``A @ B`` with numpy broadcasting over leading dims.
    """
    A_arr = _as_array(A, "A")
    B_arr = _as_array(B, "B")
    _check_transform_shape(A_arr, "A")
    _check_transform_shape(B_arr, "B")
    return A_arr @ B_arr


def relative_transform(T_a, T_b) -> np.ndarray:
    """Return frame-b as seen from frame-a: ``T_a^-1 @ T_b``.

    Args:
        T_a: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.
        T_b: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Returns:
        Relative transform(s), broadcast over leading dimensions.
    """
    return compose_transform(invert_transform(T_a), T_b)


def mat_to_rot6d_xyz(T) -> np.ndarray:
    """Encode transform(s) as ``[R[:, 0], R[:, 1], xyz]``.

    Args:
        T: Array-like with shape ``(4, 4)`` or ``(..., 4, 4)``.

    Returns:
        Array with shape ``(9,)`` or ``(..., 9)``. The first six values are
        the first two rotation-matrix columns, followed by xyz translation.
    """
    T_arr = _as_array(T, "T")
    _check_transform_shape(T_arr, "T")
    r0 = T_arr[..., :3, 0]
    r1 = T_arr[..., :3, 1]
    xyz = T_arr[..., :3, 3]
    return np.concatenate([r0, r1, xyz], axis=-1)
