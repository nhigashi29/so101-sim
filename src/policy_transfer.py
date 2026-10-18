"""帰還タスクへの観測追加。既習入力の重みと正規化を保持する。"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from return_policy import ReturnHomePolicy


def transfer_to_return(
    source: Path,
    raw: Any,
    learning_rate: float,
    target_kl: float,
    entropy: float,
) -> tuple[VecNormalize, PPO]:
    # These are the same trusted local SB3 checkpoint files used by PPO.load.
    with (source / "normalize.pkl").open("rb") as stream:
        env = pickle.load(stream)
    old_size = env.observation_space.shape[0]
    new_size = raw.observation_space.shape[0]
    if old_size not in (69, 77) or new_size != 77:
        raise ValueError(f"帰還への転移は69/77→77次元だけに対応: {old_size}→{new_size}")
    env.obs_rms.mean = np.r_[env.obs_rms.mean, np.zeros(new_size - old_size)]
    env.obs_rms.var = np.r_[env.obs_rms.var, np.ones(new_size - old_size)]
    env.observation_space = raw.observation_space
    env.set_venv(raw)
    env.training = True
    old = PPO.load(str(source / "policy.zip"), device="cpu")
    model = PPO(
        ReturnHomePolicy,
        env,
        learning_rate=learning_rate,
        target_kl=target_kl,
        ent_coef=entropy,
        n_steps=old.n_steps,
        batch_size=old.batch_size,
        n_epochs=old.n_epochs,
        gamma=old.gamma,
        gae_lambda=old.gae_lambda,
        clip_range=old.clip_range,
        clip_range_vf=old.clip_range_vf,
        vf_coef=old.vf_coef,
        max_grad_norm=old.max_grad_norm,
        policy_kwargs=old.policy_kwargs,
        seed=old.seed,
        device="cpu",
        verbose=1,
    )
    previous = old.policy.state_dict()
    expanded = model.policy.state_dict()
    for name, value in expanded.items():
        if name.startswith(("action_net.retreat.", "action_net.home.")):
            continue
        previous_name = name.replace("action_net.pick.", "action_net.")
        original = previous[previous_name]
        if value.shape == original.shape:
            expanded[name] = original
        elif (
            name
            in (
                "mlp_extractor.policy_net.0.weight",
                "mlp_extractor.value_net.0.weight",
            )
            and value.shape == (original.shape[0], new_size)
            and original.shape[1] == old_size
        ):
            value.zero_()
            value[:, :old_size] = original
        else:
            raise ValueError(f"移行できない方策パラメーター: {name}")
    model.policy.load_state_dict(expanded)
    model.num_timesteps = old.num_timesteps
    # Optimizer moments cannot be reused after changing the input tensor shape.
    # Fresh moments, old network weights, unchanged log_std and old obs statistics.
    with torch.no_grad():
        probe = torch.zeros((1, new_size), device=model.device)
        if not torch.isfinite(model.policy(probe)[0]).all():
            raise RuntimeError("転移した方策の初期推論が有限値ではありません。")
    return env, model
