"""Minimal tests for bimanual relation graph observations.

Run with:
    python -m robotwin_birelpose.tests.test_relation_graph
"""

import numpy as np

from robotwin_birelpose.relation_graph import build_bimanual_relation_obs, get_relation_obs_dim


def _pose(xyz):
    return np.array([xyz[0], xyz[1], xyz[2], 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_relation_obs_dims():
    """Relation graph dimension matches the requested edge set."""
    left_pose = _pose([0.1, 0.0, 0.2])
    right_pose = _pose([-0.1, 0.0, 0.2])
    object_pose = _pose([0.0, 0.0, 0.0])
    target_pose = _pose([0.0, 0.2, 0.0])

    default_obs = build_bimanual_relation_obs(
        left_pose,
        right_pose,
        object_pose,
        left_gripper=0.25,
        right_gripper=0.75,
    )
    assert default_obs.shape == (29,)
    assert get_relation_obs_dim() == 29

    no_lr_obs = build_bimanual_relation_obs(
        left_pose,
        right_pose,
        object_pose,
        left_gripper=0.25,
        right_gripper=0.75,
        use_left_right_relation=False,
    )
    assert no_lr_obs.shape == (20,)
    assert get_relation_obs_dim(use_left_right_relation=False) == 20

    target_obs = build_bimanual_relation_obs(
        left_pose,
        right_pose,
        object_pose,
        left_gripper=0.25,
        right_gripper=0.75,
        target_pose=target_pose,
        use_target_relation=True,
    )
    assert target_obs.shape == (38,)
    assert get_relation_obs_dim(use_target_relation=True) == 38


def test_grippers_are_appended():
    """The final two observation values are left and right gripper states."""
    obs = build_bimanual_relation_obs(
        _pose([0.0, 0.0, 0.0]),
        _pose([1.0, 0.0, 0.0]),
        _pose([0.0, 1.0, 0.0]),
        left_gripper=0.1,
        right_gripper=0.9,
    )
    np.testing.assert_allclose(obs[-2:], [0.1, 0.9], atol=1e-7)


def main():
    test_relation_obs_dims()
    test_grippers_are_appended()
    print("test_relation_graph: all tests passed")


if __name__ == "__main__":
    main()
