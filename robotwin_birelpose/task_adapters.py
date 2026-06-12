"""Task-specific object/target pose adapters for RoboTwin runtime tasks.

RoboTwin demonstrations do not expose object pose through a uniform HDF5 field.
This module extracts poses from a live task instance after ``setup_demo`` using
task-specific actor attribute names verified from ``robotwin/envs/*.py``.

All returned poses use ``xyz + qw qx qy qz``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TASK_OBJECT_ADAPTERS = {
    "place_a2b_right": {
        "object_attrs": ["object"],
        "target_attrs": ["target_object"],
    },
    "pick_dual_bottles": {
        "object_attrs": ["bottle1", "bottle2"],
        "target_attrs": [],
        "target_pose_attrs": ["left_target_pose", "right_target_pose"],
    },
    "place_can_basket": {
        "object_attrs": ["can"],
        "target_attrs": ["basket"],
    },
    "move_can_pot": {
        "object_attrs": ["can"],
        "target_attrs": ["pot"],
        "target_pose_attrs": ["target_pose"],
    },
    "place_container_plate": {
        "object_attrs": ["container"],
        "target_attrs": ["plate"],
    },
    "place_dual_shoes": {
        "object_attrs": ["left_shoe", "right_shoe"],
        "target_attrs": ["shoe_box"],
    },
    "handover_block": {
        "object_attrs": ["box"],
        "target_attrs": ["target_box"],
        "target_pose_attrs": ["block_middle_pose"],
    },
}


def _as_pose7(value: Any, name: str) -> np.ndarray:
    """Convert a pose-like value to shape ``(7,)`` in ``xyz + qw qx qy qz``."""
    if hasattr(value, "p") and hasattr(value, "q"):
        return pose_from_sapien_pose(value)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"{name} must be pose7 with shape (7,) or a pose with .p/.q; got {arr.shape}")
    return arr


def _select_name(names: list[str], index: int, kind: str, task_name: str) -> str:
    if not names:
        raise ValueError(f"Task {task_name!r} has no configured {kind} entries")
    if index < 0 or index >= len(names):
        raise IndexError(
            f"{kind} index {index} is out of range for task {task_name!r}; "
            f"available {kind}s: {names}"
        )
    return names[index]


def _get_adapter(task_name: str) -> dict:
    try:
        return TASK_OBJECT_ADAPTERS[task_name]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported task {task_name!r}. Add it to TASK_OBJECT_ADAPTERS before extracting poses."
        ) from exc


def pose_from_sapien_pose(pose) -> np.ndarray:
    """Convert a Sapien-compatible pose to ``xyz + qw qx qy qz``.

    Args:
        pose: Object with ``pose.p`` and ``pose.q`` attributes. RoboTwin/Sapien
            uses quaternion order ``qw qx qy qz``.

    Returns:
        Numpy array with shape ``(7,)``.
    """
    if not hasattr(pose, "p") or not hasattr(pose, "q"):
        raise TypeError("pose must provide .p and .q attributes")
    p = np.asarray(pose.p, dtype=np.float64)
    q = np.asarray(pose.q, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"pose.p must have shape (3,); got {p.shape}")
    if q.shape != (4,):
        raise ValueError(f"pose.q must have shape (4,) in qw qx qy qz order; got {q.shape}")
    return np.concatenate([p, q], axis=0)


def get_attr_pose(task, attr_name: str) -> np.ndarray:
    """Read ``task.<attr_name>.get_pose()`` and return pose7.

    Args:
        task: RoboTwin task instance.
        attr_name: Actor attribute name such as ``"object"`` or ``"bottle1"``.

    Returns:
        Pose array with shape ``(7,)`` in ``xyz + qw qx qy qz``.

    Raises:
        AttributeError: If the task does not have ``attr_name`` or the actor
            does not provide ``get_pose``.
    """
    if not hasattr(task, attr_name):
        raise AttributeError(f"Task {type(task).__name__!r} has no attribute {attr_name!r}")
    actor = getattr(task, attr_name)
    if not hasattr(actor, "get_pose"):
        raise AttributeError(f"Task attribute {attr_name!r} does not provide get_pose()")
    return pose_from_sapien_pose(actor.get_pose())


def extract_object_poses_from_task(
    task,
    task_name: str,
    object_index: int | None = 0,
    return_all: bool = True,
) -> dict:
    """Extract object pose(s) from a configured RoboTwin task instance.

    Args:
        task: RoboTwin task instance after ``setup_demo``.
        task_name: RoboTwin task name used to look up ``TASK_OBJECT_ADAPTERS``.
        object_index: Which object to expose as ``object_pose`` when multiple
            objects exist. If ``None``, the first object is used for the scalar
            ``object_pose`` while all objects remain in ``all_object_poses``.
        return_all: Include ``all_object_poses`` in the returned dict.

    Returns:
        Dict containing ``object_pose`` with shape ``(7,)``, ``object_name``,
        and optionally ``all_object_poses``.
    """
    adapter = _get_adapter(task_name)
    object_attrs = list(adapter.get("object_attrs", []))
    select_index = 0 if object_index is None else object_index
    object_name = _select_name(object_attrs, select_index, "object", task_name)
    all_object_poses = {name: get_attr_pose(task, name) for name in object_attrs}

    result = {
        "object_pose": all_object_poses[object_name],
        "object_name": object_name,
    }
    if return_all:
        result["all_object_poses"] = all_object_poses
    return result


def extract_target_pose_from_task(
    task,
    task_name: str,
    target_index: int = 0,
) -> dict | None:
    """Extract an optional target pose from a configured RoboTwin task.

    Target pose is optional. If ``target_attrs`` are configured, the target is
    read via actor ``get_pose()``. Otherwise, ``target_pose_attrs`` are read
    directly and may be pose7 lists/arrays or Sapien-compatible pose objects.

    Args:
        task: RoboTwin task instance after ``setup_demo``.
        task_name: RoboTwin task name used to look up ``TASK_OBJECT_ADAPTERS``.
        target_index: Which configured target entry to read.

    Returns:
        ``{"target_pose": pose7, "target_name": name}`` or ``None`` if no
        target rule exists for this task.
    """
    adapter = _get_adapter(task_name)
    target_attrs = list(adapter.get("target_attrs", []))
    target_pose_attrs = list(adapter.get("target_pose_attrs", []))

    if target_attrs:
        target_name = _select_name(target_attrs, target_index, "target", task_name)
        return {
            "target_pose": get_attr_pose(task, target_name),
            "target_name": target_name,
        }

    if target_pose_attrs:
        target_name = _select_name(target_pose_attrs, target_index, "target_pose", task_name)
        if not hasattr(task, target_name):
            raise AttributeError(f"Task {type(task).__name__!r} has no attribute {target_name!r}")
        return {
            "target_pose": _as_pose7(getattr(task, target_name), target_name),
            "target_name": target_name,
        }

    return None


def print_task_adapter_summary() -> None:
    """Print a compact summary of supported task adapters."""
    for task_name, adapter in TASK_OBJECT_ADAPTERS.items():
        objects = ", ".join(adapter.get("object_attrs", [])) or "-"
        targets = ", ".join(adapter.get("target_attrs", [])) or "-"
        target_poses = ", ".join(adapter.get("target_pose_attrs", [])) or "-"
        print(f"{task_name}: objects=[{objects}], targets=[{targets}], target_pose_attrs=[{target_poses}]")
