"""Minimal tests for SE(3) utilities.

Run with:
    python -m robotwin_birelpose.tests.test_se3_utils
"""

import numpy as np

from robotwin_birelpose.se3_utils import (
    mat_to_pose7_wxyz,
    normalize_quat_wxyz,
    pose7_wxyz_to_mat,
    relative_transform,
)


def test_pose7_round_trip():
    """pose7 -> matrix -> pose7 preserves xyz and normalized wxyz quaternion."""
    pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    T = pose7_wxyz_to_mat(pose)
    recovered = mat_to_pose7_wxyz(T)

    np.testing.assert_allclose(recovered[:3], pose[:3], atol=1e-8)
    np.testing.assert_allclose(recovered[3:], normalize_quat_wxyz(pose[3:]), atol=1e-8)


def test_relative_transform_translation():
    """Identity frame sees a translated frame at that translation."""
    T_a = np.eye(4)
    T_b = np.eye(4)
    T_b[:3, 3] = [1.0, 2.0, 3.0]

    T_rel = relative_transform(T_a, T_b)
    np.testing.assert_allclose(T_rel[:3, 3], [1.0, 2.0, 3.0], atol=1e-8)


def test_quaternion_order_wxyz_not_xyzw():
    """A 180 degree x-rotation is encoded as qw qx qy qz = [0, 1, 0, 0]."""
    pose = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    T = pose7_wxyz_to_mat(pose)
    expected_R = np.diag([1.0, -1.0, -1.0])

    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-8)


def main():
    test_pose7_round_trip()
    test_relative_transform_translation()
    test_quaternion_order_wxyz_not_xyzw()
    print("test_se3_utils: all tests passed")


if __name__ == "__main__":
    main()
