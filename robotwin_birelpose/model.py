from __future__ import annotations

import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy


class BiRelPoseDiffusionPolicy(nn.Module):
    """Pose-only bimanual relative-pose diffusion policy.

    Inputs:
        obs: ``[B, n_obs_steps, obs_dim]``
        action: ``[B, horizon, action_dim]``

    The wrapper flattens the relative-pose observation window, encodes it with
    an MLP, and reuses PoseInsert's ``DiffusionUNetPolicy`` as the action
    trajectory decoder. The wrapped decoder is kept at one conditioning step
    because PoseInsert's original diffusion loss expects a single readout per
    batch item.
    """

    def __init__(
        self,
        obs_dim: int = 29,
        action_dim: int = 16,
        horizon: int = 16,
        n_obs_steps: int = 2,
        obs_feature_dim: int = 128,
        hidden_dim: int = 256,
        prediction: str = "sample",
        down_dims=(128, 256),
        num_inference_steps: int = 20,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_feature_dim = int(obs_feature_dim)

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim * self.n_obs_steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.obs_feature_dim),
            nn.SiLU(),
        )
        self.action_decoder = DiffusionUNetPolicy(
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=1,
            obs_feature_dim=self.obs_feature_dim,
            prediction=prediction,
            down_dims=down_dims,
            num_inference_steps=num_inference_steps,
        )

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode ``obs`` of shape ``[B, n_obs_steps, obs_dim]``."""

        if obs.ndim != 3:
            raise ValueError(f"obs must have shape [B, n_obs_steps, obs_dim], got {tuple(obs.shape)}")
        if obs.shape[1] != self.n_obs_steps or obs.shape[2] != self.obs_dim:
            raise ValueError(
                f"obs must have shape [B, {self.n_obs_steps}, {self.obs_dim}], "
                f"got {tuple(obs.shape)}"
            )
        flat_obs = obs.reshape(obs.shape[0], self.obs_dim * self.n_obs_steps)
        return self.obs_encoder(flat_obs)

    def forward_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return scalar diffusion training loss for a normalized batch."""

        if action.ndim != 3:
            raise ValueError(f"action must have shape [B, horizon, action_dim], got {tuple(action.shape)}")
        if action.shape[1] != self.horizon or action.shape[2] != self.action_dim:
            raise ValueError(
                f"action must have shape [B, {self.horizon}, {self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        readout = self.encode_obs(obs)
        return self.action_decoder.compute_loss(readout, action)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample an action trajectory with shape ``[B, horizon, action_dim]``."""

        readout = self.encode_obs(obs)
        return self.action_decoder.predict_action(readout)


class BiRelPoseMlpPolicy(nn.Module):
    """Small fallback MLP policy for pipeline smoke tests only.

    This class is not the final BiRelPoseDP method; it is useful if diffusion
    dependencies are unavailable while debugging the dataset/training loop.
    """

    def __init__(
        self,
        obs_dim: int = 29,
        action_dim: int = 16,
        horizon: int = 16,
        n_obs_steps: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim * self.n_obs_steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.horizon * self.action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError(f"obs must have shape [B, n_obs_steps, obs_dim], got {tuple(obs.shape)}")
        pred = self.net(obs.reshape(obs.shape[0], -1))
        return pred.reshape(obs.shape[0], self.horizon, self.action_dim)

    def forward_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.mean((self.forward(obs) - action) ** 2)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
