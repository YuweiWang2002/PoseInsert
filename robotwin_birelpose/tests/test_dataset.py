from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from robotwin_birelpose.dataset import (
    BiRelPoseNpzDataset,
    compute_normalizer,
    load_normalizer,
    normalize_batch,
    save_normalizer,
)


def test_dataset_windows_and_normalizer():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode0.npz"
        rel_obs = np.arange(30 * 29, dtype=np.float32).reshape(30, 29)
        action_ee = np.arange(30 * 16, dtype=np.float32).reshape(30, 16)
        np.savez(path, rel_obs=rel_obs, action_ee=action_ee)

        dataset = BiRelPoseNpzDataset(tmpdir, n_obs_steps=2, horizon=16)
        assert len(dataset) == 14

        sample = dataset[0]
        assert sample["obs"].shape == (2, 29)
        assert sample["action"].shape == (16, 16)
        assert sample["t"] == 1
        np.testing.assert_allclose(sample["obs"], rel_obs[0:2])
        np.testing.assert_allclose(sample["action"], action_ee[1:17])

        normalizer = compute_normalizer(dataset)
        assert normalizer["obs_mean"].shape == (29,)
        assert normalizer["obs_std"].shape == (29,)
        assert normalizer["action_mean"].shape == (16,)
        assert normalizer["action_std"].shape == (16,)

        norm_path = Path(tmpdir) / "normalizer.npz"
        save_normalizer(norm_path, normalizer)
        loaded = load_normalizer(norm_path)
        np.testing.assert_allclose(loaded["obs_mean"], normalizer["obs_mean"])

        batch = {
            "obs": np.stack([sample["obs"], sample["obs"]], axis=0),
            "action": np.stack([sample["action"], sample["action"]], axis=0),
        }
        normalized = normalize_batch(batch, normalizer, to_torch=False)
        assert normalized["obs"].shape == (2, 2, 29)
        assert normalized["action"].shape == (2, 16, 16)


def main():
    test_dataset_windows_and_normalizer()
    print("test_dataset: all tests passed")


if __name__ == "__main__":
    main()
