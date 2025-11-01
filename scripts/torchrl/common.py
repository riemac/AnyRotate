# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchRL 训练脚本之间共享的辅助工具函数。

本模块提供构建 TorchRL 网络模块和优化器的工具函数，遵循 TorchRL 的组件化设计。

Components:
    - **Actor Construction**: ``build_actor()`` - 构建概率策略网络（ProbabilisticActor）
    - **Critic Construction**: ``build_critic()`` - 构建状态价值网络（ValueOperator）
    - **Optimizer Setup**: ``prepare_optimizer()`` - 配置 Adam 优化器
    - **Utility Functions**: 规格选择、维度展平、激活函数映射等

Network Architecture:
    默认网络结构遵循标准 Actor-Critic 架构：
    
    **Actor (ProbabilisticActor)**::
    
        Observation → Flatten → MLP([256, 256]) → 2×action_dim → (loc, scale)
                                                                        ↓
                                                            TanhNormal Distribution
                                                                        ↓
                                                                     Action
    
    **Critic (ValueOperator)**::
    
        Observation → Flatten → MLP([256, 256]) → 1 → State Value

Note:
    - 默认使用 ReLU 激活函数
    - Actor 输出经过 TanhNormal 分布以限制动作范围
    - 支持通过配置字典自定义网络结构（隐藏层大小、激活函数等）

Example:
    构建自定义网络::
    
        actor_cfg = {
            "hidden_sizes": [512, 512, 256],
            "activation": "elu",
            "scale_mapping": "biased_softplus_1.0"
        }
        actor = build_actor(obs_key, action_key, action_spec, obs_dim, actor_cfg, device)

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.utils import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal


def split_key(key: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return key
    if isinstance(key, list):
        return tuple(key)
    return tuple(key.split(".")) if isinstance(key, str) else (key,)


def get_activation(name: str) -> type[nn.Module]:
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    return mapping.get(name.lower(), nn.ReLU)


def select_spec(composite, key: tuple[str, ...]):
    spec = composite
    for part in key:
        if part not in spec.keys():  # type: ignore[attr-defined]
            available = ", ".join(str(k) for k in spec.keys())  # type: ignore[attr-defined]
            raise KeyError(f"Unable to resolve key '{'.'.join(key)}'. Available keys: {available}")
        spec = spec[part]
    return spec


def flatten_size(shape: torch.Size | tuple[int, ...]) -> int:
    return int(np.prod(tuple(shape))) if len(shape) > 0 else 1


def build_actor(
    obs_key: tuple[str, ...],
    action_key: tuple[str, ...],
    action_spec,
    obs_dim: int,
    cfg: dict,
    device: torch.device,
) -> ProbabilisticActor:
    hidden_sizes = cfg.get("hidden_sizes", [256, 256])
    activation = get_activation(cfg.get("activation", "relu"))
    scale_mapping = cfg.get("scale_mapping", "biased_softplus_1.0")
    scale_lb = cfg.get("scale_lb", 1e-4)
    action_dim = flatten_size(action_spec.shape)
    in_features = cfg.get("in_features", obs_dim)

    actor_backbone = nn.Sequential(
        nn.Flatten(),
        MLP(
            in_features=in_features,
            num_cells=hidden_sizes,
            out_features=2 * action_dim,
            activation_class=activation,
        ),
        NormalParamExtractor(scale_mapping=scale_mapping, scale_lb=scale_lb),
    )
    actor_module = TensorDictModule(
        actor_backbone,
        in_keys=[obs_key],
        out_keys=["loc", "scale"],
    )

    low = action_spec.space.low if hasattr(action_spec, "space") else action_spec.minimum
    high = action_spec.space.high if hasattr(action_spec, "space") else action_spec.maximum
    low = torch.as_tensor(low, device=device)
    high = torch.as_tensor(high, device=device)

    policy = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=[action_key],
        spec=action_spec,
        default_interaction_type=ExplorationType.RANDOM,
        distribution_class=TanhNormal,
        distribution_kwargs={"min": low, "max": high},
        return_log_prob=True,
    )
    policy.to(device)
    return policy


def build_critic(obs_key: tuple[str, ...], obs_dim: int, cfg: dict, device: torch.device) -> ValueOperator:
    hidden_sizes = cfg.get("hidden_sizes", [256, 256])
    activation = get_activation(cfg.get("activation", "relu"))
    in_features = cfg.get("in_features", obs_dim)

    critic_backbone = nn.Sequential(
        nn.Flatten(),
        MLP(
            in_features=in_features,
            num_cells=hidden_sizes,
            out_features=1,
            activation_class=activation,
        ),
    )

    critic = ValueOperator(
        TensorDictModule(critic_backbone, in_keys=[obs_key], out_keys=["state_value"]),
        out_keys=["state_value"],
    )
    critic.to(device)
    return critic


def prepare_optimizer(parameters, optimizer_cfg: dict):
    lr = optimizer_cfg.get("lr", 3e-4)
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    return torch.optim.Adam(parameters, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
