"""Minimal tests for the RoboTwin HDF5 endpose reader.

Run with:
    python -m robotwin_birelpose.tests.test_hdf5_endpose_reader
"""

import tempfile
from pathlib import Path

import numpy as np

from robotwin_birelpose.hdf5_endpose_reader import load_endpose_episode, make_ee_action_array


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("test_hdf5_endpose_reader requires h5py") from exc
    return h5py


def test_load_endpose_episode():
    """Reader loads RoboTwin /endpose fields and preserves wxyz pose layout."""
    h5py = _require_h5py()
    left = np.array(
        [
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            [-0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            [-0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    left_gripper = np.array([0.0, 1.0], dtype=np.float32)
    right_gripper = np.array([1.0, 0.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode0.hdf5"
        with h5py.File(path, "w") as root:
            group = root.create_group("endpose")
            group.create_dataset("left_endpose", data=left)
            group.create_dataset("right_endpose", data=right)
            group.create_dataset("left_gripper", data=left_gripper)
            group.create_dataset("right_gripper", data=right_gripper)

        episode = load_endpose_episode(path)

    assert episode.num_frames == 2
    np.testing.assert_allclose(episode.left_ee_pose, left)
    np.testing.assert_allclose(episode.right_ee_pose, right)
    np.testing.assert_allclose(episode.left_gripper, left_gripper)
    np.testing.assert_allclose(episode.right_gripper, right_gripper)

    ee_actions = episode.as_ee_action_array()
    assert ee_actions.shape == (2, 16)
    np.testing.assert_allclose(ee_actions[0, :7], left[0])
    np.testing.assert_allclose(ee_actions[0, 7], left_gripper[0])
    np.testing.assert_allclose(ee_actions[0, 8:15], right[0])
    np.testing.assert_allclose(ee_actions[0, 15], right_gripper[0])


def test_make_ee_action_array_shape():
    """Packing endposes creates RoboTwin ee action dim 16."""
    left = np.zeros((3, 7), dtype=np.float32)
    right = np.ones((3, 7), dtype=np.float32)
    left[:, 3] = 1.0
    right[:, 3] = 1.0
    left_gripper = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    right_gripper = np.linspace(1.0, 0.0, 3, dtype=np.float32)

    action = make_ee_action_array(left, right, left_gripper, right_gripper)

    assert action.shape == (3, 16)
    np.testing.assert_allclose(action[:, 7], left_gripper)
    np.testing.assert_allclose(action[:, 15], right_gripper)


def main():
    test_load_endpose_episode()
    test_make_ee_action_array_shape()
    print("test_hdf5_endpose_reader: all tests passed")


if __name__ == "__main__":
    main()
