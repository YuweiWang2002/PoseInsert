"""Tests for object-relative action conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from robotwin_birelpose.relative_action import (
    absolute_ee_action_to_relative_action,
    convert_npz_absolute_to_relative_output,
    relative_action_to_absolute_ee_action,
)


def _quat_dot_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.abs(np.sum(a * b, axis=-1))


def _assert_action_round_trip(action_ee, object_pose, atol=1e-5):
    rel_action = absolute_ee_action_to_relative_action(action_ee, object_pose)
    recon = relative_action_to_absolute_ee_action(rel_action, object_pose)
    assert rel_action.shape[-1] == 20
    assert recon.shape == np.asarray(action_ee).shape
    np.testing.assert_allclose(recon[..., :3], np.asarray(action_ee)[..., :3], atol=atol)
    np.testing.assert_allclose(recon[..., 8:11], np.asarray(action_ee)[..., 8:11], atol=atol)
    np.testing.assert_allclose(_quat_dot_abs(recon[..., 3:7], np.asarray(action_ee)[..., 3:7]), 1.0, atol=atol)
    np.testing.assert_allclose(_quat_dot_abs(recon[..., 11:15], np.asarray(action_ee)[..., 11:15]), 1.0, atol=atol)
    np.testing.assert_allclose(recon[..., 7], np.asarray(action_ee)[..., 7], atol=atol)
    np.testing.assert_allclose(recon[..., 15], np.asarray(action_ee)[..., 15], atol=atol)


def test_identity_object_pose_round_trip():
    object_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action_ee = np.array(
        [
            0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.25,
            -0.2, 0.1, 0.4, 0.9238795, 0.0, 0.3826834, 0.0, 0.75,
        ],
        dtype=np.float32,
    )
    _assert_action_round_trip(action_ee, object_pose)


def test_translated_object_pose_round_trip():
    object_pose = np.array([0.5, -0.25, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action_ee = np.array(
        [
            0.6, -0.1, 0.4, 0.9238795, 0.3826834, 0.0, 0.0, 0.0,
            0.2, 0.3, 0.5, 0.7071068, 0.0, 0.0, 0.7071068, 1.0,
        ],
        dtype=np.float32,
    )
    _assert_action_round_trip(action_ee, object_pose)


def test_batched_conversion():
    T = 5
    object_pose = np.tile(np.array([0.2, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
    action_ee = np.tile(
        np.array(
            [
                0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.1,
                -0.1, 0.3, 0.45, 1.0, 0.0, 0.0, 0.0, 0.9,
            ],
            dtype=np.float32,
        ),
        (T, 1),
    )
    action_ee[:, 0] += np.linspace(0.0, 0.04, T)
    action_ee[:, 8] -= np.linspace(0.0, 0.04, T)

    rel_action = absolute_ee_action_to_relative_action(action_ee, object_pose)
    recon = relative_action_to_absolute_ee_action(rel_action, object_pose)
    assert rel_action.shape == (T, 20)
    assert recon.shape == (T, 16)
    _assert_action_round_trip(action_ee, object_pose)


def test_npz_conversion_fake():
    T = 30
    rel_obs = np.zeros((T, 29), dtype=np.float32)
    object_pose = np.tile(np.array([0.2, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
    action_ee = np.tile(
        np.array(
            [
                0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.1,
                -0.1, 0.3, 0.45, 1.0, 0.0, 0.0, 0.0, 0.9,
            ],
            dtype=np.float32,
        ),
        (T, 1),
    )

    with tempfile.TemporaryDirectory() as tmp:
        input_npz = Path(tmp) / "episode0.npz"
        output_npz = Path(tmp) / "relout" / "episode0.npz"
        np.savez_compressed(input_npz, rel_obs=rel_obs, action_ee=action_ee, object_pose=object_pose)
        convert_npz_absolute_to_relative_output(input_npz, output_npz)
        converted = np.load(output_npz)
        assert converted["rel_obs"].shape == (T, 29)
        assert converted["rel_action"].shape == (T, 20)
        assert converted["action_ee"].shape == (T, 16)
        assert converted["object_pose"].shape == (T, 7)


if __name__ == "__main__":
    test_identity_object_pose_round_trip()
    test_translated_object_pose_round_trip()
    test_batched_conversion()
    test_npz_conversion_fake()
    print("test_relative_action passed")
