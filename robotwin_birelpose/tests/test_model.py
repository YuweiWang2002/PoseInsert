import torch

from robotwin_birelpose.model import BiRelPoseDiffusionPolicy, BiRelPoseMlpPolicy


def _check_diffusion_policy_shapes(action_dim: int):
    torch.manual_seed(0)
    model = BiRelPoseDiffusionPolicy(
        obs_dim=29,
        action_dim=action_dim,
        horizon=16,
        n_obs_steps=2,
        obs_feature_dim=32,
        hidden_dim=64,
        down_dims=(32, 64),
        num_inference_steps=2,
    )
    obs = torch.randn(4, 2, 29)
    action = torch.randn(4, 16, action_dim)
    loss = model.forward_loss(obs, action)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    sample = model.sample_action(obs)
    assert sample.shape == (4, 16, action_dim)


def test_diffusion_policy_shapes():
    _check_diffusion_policy_shapes(action_dim=16)
    _check_diffusion_policy_shapes(action_dim=20)


def test_mlp_policy_shapes():
    model = BiRelPoseMlpPolicy(obs_dim=29, action_dim=16, horizon=16, n_obs_steps=2)
    obs = torch.randn(4, 2, 29)
    action = torch.randn(4, 16, 16)
    loss = model.forward_loss(obs, action)
    assert loss.ndim == 0
    sample = model.sample_action(obs)
    assert sample.shape == (4, 16, 16)


def main():
    test_diffusion_policy_shapes()
    test_mlp_policy_shapes()
    print("test_model: all tests passed")


if __name__ == "__main__":
    main()
