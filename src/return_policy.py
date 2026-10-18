"""既習の配置出力を保持し、退避・帰還の出力をPPOで学ぶ単一方策。"""

from collections.abc import Callable
from typing import cast

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class ReturnMlpExtractor(MlpExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_size = args[0]
        self.latent_dim_pi += self.observation_size + 2
        self.policy_net.requires_grad_(False)

    def forward_actor(self, features):
        latent = super().forward_actor(features)
        # Flags are binary before VecNormalize. Their normalized value is
        # positive iff the original flag is one (running means stay in [0, 1)).
        flags = (features[:, 69:71] > 0).to(latent.dtype)
        return torch.cat([latent, features, flags], dim=1)


class PhaseActionNet(nn.Module):
    def __init__(self, latent_size, observation_size, action_size):
        super().__init__()
        self.latent_size = latent_size
        self.pick = nn.Linear(latent_size, action_size)
        self.pick.requires_grad_(False)
        self.retreat = nn.Sequential(
            nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, action_size)
        )
        self.home = nn.Sequential(
            nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, action_size)
        )
        nn.init.orthogonal_(self.pick.weight, gain=0.01)
        nn.init.zeros_(self.pick.bias)
        for layer in (self.retreat, self.home):
            output = layer[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, latent):
        pick_features = latent[:, : self.latent_size]
        features = latent[:, self.latent_size : -2]
        retreat, home = latent[:, -2:-1], latent[:, -1:]
        return (
            (1 - retreat - home) * self.pick(pick_features)
            + retreat * self.retreat(features)
            + home * self.home(features)
        )


class ReturnHomePolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = ReturnMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        shape = self.action_space.shape
        assert shape is not None
        self.action_net = PhaseActionNet(
            self.mlp_extractor.latent_dim_pi - self.features_dim - 2,
            self.features_dim,
            shape[0],
        ).to(self.device)
        optimizer_class = cast(
            Callable[..., torch.optim.Optimizer], self.optimizer_class
        )
        self.optimizer = optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
