"""Minimal tests for BiRelPose conversion helpers.

Run with:
    python -m robotwin_birelpose.tests.test_conversion
"""

from pathlib import Path
import tempfile

import numpy as np

from robotwin_birelpose.conversion import convert_episode_to_birelpose


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("test_conversion requires h5py") from exc
    return h5py


def test_convert_episode_to_birelpose_shapes():
    """Fake HDF5 endpose + object pose converts to rel_obs and ee actions."""
    h5py = _require_h5py()
    T = 4
    left = np.tile(np.array([[0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (T, 1))
    right = np.tile(np.array([[-0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (T, 1))
    left[:, 0] += np.arange(T) * 0.01
    right[:, 0] -= np.arange(T) * 0.01
    left_gripper = np.linspace(0.0, 1.0, T, dtype=np.float32)
    right_gripper = np.linspace(1.0, 0.0, T, dtype=np.float32)
    object_pose = np.tile(np.array([[0.0, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (T, 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode0.hdf5"
        with h5py.File(path, "w") as root:
            group = root.create_group("endpose")
            group.create_dataset("left_endpose", data=left)
            group.create_dataset("right_endpose", data=right)
            group.create_dataset("left_gripper", data=left_gripper)
            group.create_dataset("right_gripper", data=right_gripper)

        converted = convert_episode_to_birelpose(path, object_pose=object_pose)

    assert converted["rel_obs"].shape == (T, 29)
    assert converted["action_ee"].shape == (T, 16)
    assert converted["left_endpose"].shape == (T, 7)
    assert converted["right_endpose"].shape == (T, 7)
    assert converted["left_gripper"].shape == (T, 1)
    assert converted["right_gripper"].shape == (T, 1)
    assert converted["object_pose"].shape == (T, 7)
    np.testing.assert_allclose(converted["object_pose"][:, 3], 1.0)


def main():
    test_convert_episode_to_birelpose_shapes()
    print("test_conversion: all tests passed")


if __name__ == "__main__":
    main()
