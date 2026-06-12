"""Bimanual relative-pose observation construction for RoboTwin.

The module intentionally does not read RoboTwin HDF5 files or assume object
poses are present in demonstrations. It only consumes explicit pose7 inputs
using the convention ``xyz + qw qx qy qz``.
"""

from __future__ import annotations

import numpy as np

from .se3_utils import mat_to_rot6d_xyz, pose7_wxyz_to_mat, relative_transform


POSE_REPR_DIMS = {
    "rot6d_xyz": 9,
}


def _encode_pose_relation(T_relation: np.ndarray, pose_repr: str) -> np.ndarray:
    if pose_repr == "rot6d_xyz":
        return mat_to_rot6d_xyz(T_relation)
    raise ValueError(f"Unsupported pose_repr {pose_repr!r}; expected one of {sorted(POSE_REPR_DIMS)}")


def get_relation_obs_dim(
    use_left_right_relation: bool = True,
    use_target_relation: bool = False,
    pose_repr: str = "rot6d_xyz",
) -> int:
    """Return the 1D observation dimension for the configured relation graph.

    Args:
        use_left_right_relation: Include ``T_left^-1 T_right`` if true.
        use_target_relation: Include ``T_target^-1 T_obj`` if true.
        pose_repr: Relation representation. Currently only ``"rot6d_xyz"``
            is supported and contributes 9 dimensions per relation.

    Returns:
        Observation dimension, including two gripper scalars.
    """
    if pose_repr not in POSE_REPR_DIMS:
        raise ValueError(f"Unsupported pose_repr {pose_repr!r}; expected one of {sorted(POSE_REPR_DIMS)}")
    relation_count = 2
    if use_left_right_relation:
        relation_count += 1
    if use_target_relation:
        relation_count += 1
    return relation_count * POSE_REPR_DIMS[pose_repr] + 2


def build_bimanual_relation_obs(
    left_ee_pose,
    right_ee_pose,
    object_pose,
    left_gripper: float,
    right_gripper: float,
    target_pose=None,
    use_left_right_relation: bool = True,
    use_target_relation: bool = False,
    pose_repr: str = "rot6d_xyz",
) -> np.ndarray:
    """Build a 1D bimanual relative-pose observation.

    Args:
        left_ee_pose: Left end-effector pose with shape ``(7,)`` in
            ``xyz + qw qx qy qz`` order.
        right_ee_pose: Right end-effector pose with shape ``(7,)`` in
            ``xyz + qw qx qy qz`` order.
        object_pose: Manipulated object pose with shape ``(7,)`` in
            ``xyz + qw qx qy qz`` order. This must be supplied by a task
            adapter or replay extractor; it is not assumed to exist in HDF5.
        left_gripper: Normalized left gripper value in ``[0, 1]``.
        right_gripper: Normalized right gripper value in ``[0, 1]``.
        target_pose: Optional target/receptacle pose with shape ``(7,)`` in
            ``xyz + qw qx qy qz`` order.
        use_left_right_relation: Include ``T_left^-1 T_right`` if true.
        use_target_relation: Include ``T_target^-1 T_obj`` if true and
            ``target_pose`` is not None.
        pose_repr: Relation representation. Currently only ``"rot6d_xyz"``
            is supported.

    Returns:
        A 1D numpy array. With the default graph it has dimension 29:
        three 9D relations plus two gripper scalars.
    """
    if pose_repr not in POSE_REPR_DIMS:
        raise ValueError(f"Unsupported pose_repr {pose_repr!r}; expected one of {sorted(POSE_REPR_DIMS)}")

    T_left = pose7_wxyz_to_mat(left_ee_pose)
    T_right = pose7_wxyz_to_mat(right_ee_pose)
    T_obj = pose7_wxyz_to_mat(object_pose)

    relation_features = [
        _encode_pose_relation(relative_transform(T_obj, T_left), pose_repr),
        _encode_pose_relation(relative_transform(T_obj, T_right), pose_repr),
    ]

    if use_left_right_relation:
        relation_features.append(_encode_pose_relation(relative_transform(T_left, T_right), pose_repr))

    if use_target_relation and target_pose is not None:
        T_target = pose7_wxyz_to_mat(target_pose)
        relation_features.append(_encode_pose_relation(relative_transform(T_target, T_obj), pose_repr))

    grippers = np.array([left_gripper, right_gripper], dtype=np.float64)
    rel_obs = np.concatenate(relation_features + [grippers], axis=0)
    return rel_obs.astype(np.float32, copy=False)
