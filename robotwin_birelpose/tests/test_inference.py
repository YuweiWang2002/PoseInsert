from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from robotwin_birelpose.inference import (
    load_birelpose_policy,
    predict_action_sequence,
    sanitize_ee_action,
)
from robotwin_birelpose.model import BiRelPoseDiffusionPolicy


def _make_fake_checkpoint(path: Path):
    model = BiRelPoseDiffusionPolicy(
        obs_dim=29,
        action_dim=16,
        horizon=16,
        n_obs_steps=2,
        obs_feature_dim=32,
        hidden_dim=64,
        down_dims=(32, 64),
        num_inference_steps=2,
    )
    normalizer = {
        "obs_mean": np.zeros(29, dtype=np.float32),
        "obs_std": np.ones(29, dtype=np.float32),
        "action_mean": np.zeros(16, dtype=np.float32),
        "action_std": np.ones(16, dtype=np.float32),
    }
    config = {
        "model_type": "diffusion",
        "obs_feature_dim": 32,
        "hidden_dim": 64,
        "down_dims": [32, 64],
        "num_inference_steps": 2,
        "prediction": "sample",
    }
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": normalizer,
            "config": config,
            "obs_dim": 29,
            "action_dim": 16,
            "horizon": 16,
            "n_obs_steps": 2,
        },
        path,
    )


def test_load_predict_and_sanitize():
    with TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "latest.pt"
        _make_fake_checkpoint(ckpt_path)

        model, normalizer, metadata = load_birelpose_policy(ckpt_path, device="cpu")
        assert metadata["obs_dim"] == 29
        assert metadata["action_dim"] == 16
        assert metadata["horizon"] == 16
        assert metadata["n_obs_steps"] == 2
        assert normalizer["obs_mean"].shape == (29,)

        obs_queue = np.zeros((2, 29), dtype=np.float32)
        action_seq = predict_action_sequence(model, normalizer, obs_queue)
        assert action_seq.shape == (16, 16)

        action = action_seq[0].copy()
        action[3:7] = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action[11:15] = np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)
        action[7] = -1.0
        action[15] = 2.0
        clean = sanitize_ee_action(action)
        np.testing.assert_allclose(np.linalg.norm(clean[3:7]), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(clean[11:15]), 1.0, atol=1e-5)
        assert clean[7] == 0.0
        assert clean[15] == 1.0


def main():
    test_load_predict_and_sanitize()
    print("test_inference: all tests passed")


if __name__ == "__main__":
    main()
