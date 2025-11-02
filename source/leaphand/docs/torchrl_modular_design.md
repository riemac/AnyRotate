# TorchRL 模块化设计深度解析

> **目标读者**：熟悉 RL-Games 等一体化 RL 框架，想理解 TorchRL 组件化设计哲学的开发者

## 📋 **目录**

1. [设计哲学对比](#1-设计哲学对比)
2. [核心组件详解](#2-核心组件详解)
3. [PPO 算法实现剖析](#3-ppo-算法实现剖析)
4. [非对称 Actor-Critic 实现](#4-非对称-actor-critic-实现)
5. [与 RL-Games 的对比](#5-与-rl-games-的对比)
6. [自定义网络与算法](#6-自定义网络与算法)

---

## 1. 设计哲学对比

### RL-Games: 统一 Runner 模式

```python
# RL-Games 的封装式设计
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)  # 配置文件包含一切
runner.reset()
runner.run({"train": True, "play": False})  # 一键启动
```

**特点**：
- ✅ **开箱即用**：通过配置文件控制所有行为
- ✅ **简单易用**：无需理解内部实现
- ❌ **黑盒操作**：难以自定义训练循环
- ❌ **扩展性差**：修改算法需要改源码或继承复杂类

---

### TorchRL: 组件化搭建模式

```python
# TorchRL 的组件化设计
policy_module = build_actor(...)      # 组件1: 策略网络
value_module = build_critic(...)      # 组件2: 价值网络
loss_module = ClipPPOLoss(...)        # 组件3: 损失函数
optimizer = Adam(...)                 # 组件4: 优化器
collector = SyncDataCollector(...)    # 组件5: 数据收集器

# 手动组装训练循环
for data in collector:
    rollout = data.flatten(0, 1)
    loss_module.value_estimator(rollout)  # 计算优势函数
    for epoch in range(num_epochs):
        for batch in minibatches(rollout):
            optimizer.zero_grad()
            losses = loss_module(batch)       # 前向计算损失
            total_loss.backward()             # 反向传播
            optimizer.step()                  # 更新参数
```

**特点**：
- ✅ **完全透明**：每一步都在你的控制之下
- ✅ **极致灵活**：可以在任意位置插入自定义逻辑
- ✅ **易于调试**：可以单独测试每个组件
- ❌ **学习曲线陡峭**：需要理解每个组件的作用
- ❌ **代码量大**：需要手动编写训练循环

---

## 2. 核心组件详解

TorchRL 的模块化设计基于以下 5 大核心组件：

### 2.1 Environment (环境)

**作用**：定义 MDP (Markov Decision Process) 的状态转移逻辑

```python
from isaaclab_rl.torchrl import make_torchrl_env

# 将 Gym 环境转换为 TorchRL 环境
torchrl_env = make_torchrl_env(gym_env, device=device)
```

**关键概念**：
- **TensorDict**：TorchRL 的核心数据结构，类似 Python `dict` 但支持张量操作
- **Spec**：定义观测/动作空间的形状和范围
  ```python
  obs_spec = torchrl_env.observation_spec  # 观测空间规格
  action_spec = torchrl_env.action_spec    # 动作空间规格
  ```

---

### 2.2 Policy Module (策略模块)

**作用**：将观测映射为动作分布

```python
from torchrl.modules import ProbabilisticActor

# Actor 网络结构
actor_backbone = nn.Sequential(
    nn.Flatten(),
    MLP(in_features=obs_dim, num_cells=[256, 256], out_features=2*action_dim),
    NormalParamExtractor()  # 提取均值和标准差
)

# 包装为 TensorDictModule（处理嵌套键）
actor_module = TensorDictModule(
    actor_backbone,
    in_keys=[("policy",)],   # 输入键路径
    out_keys=[("loc", "scale")]  # 输出键路径
)

# 包装为概率策略（自动采样）
policy_module = ProbabilisticActor(
    module=actor_module,
    in_keys=[("loc", "scale")],
    out_keys=[("action",)],
    distribution_class=TanhNormal,  # 使用 TanhNormal 分布
    return_log_prob=True  # 返回 log π(a|s)
)
```

**数学原理**：

$$
\begin{aligned}
\mu, \log\sigma &= f_\theta(\mathbf{s}) \quad &\text{网络输出均值和对数标准差} \\
\mathbf{a} &\sim \text{TanhNormal}(\mu, \sigma) \quad &\text{从分布中采样动作} \\
\log \pi_\theta(\mathbf{a}|\mathbf{s}) &= \log \mathcal{N}(\mathbf{a}|\mu, \sigma) - \sum_i \log(1 - \tanh^2(a_i)) \quad &\text{对数概率密度}
\end{aligned}
$$

---

### 2.3 Value Module (价值模块)

**作用**：估计状态价值 $V(s)$

```python
from torchrl.modules import ValueOperator

# Critic 网络结构
critic_backbone = nn.Sequential(
    nn.Flatten(),
    MLP(in_features=critic_obs_dim, num_cells=[256, 256], out_features=1)
)

# 包装为价值估计器
value_module = ValueOperator(
    module=critic_backbone,
    in_keys=[("critic",)],  # Critic 可以使用不同观测
    out_keys=[("state_value",)]
)
```

**数学原理**：

$$
V_\phi(\mathbf{s}) = f_\phi(\mathbf{s}) \quad \text{估计状态价值}
$$

---

### 2.4 Loss Module (损失模块)

**作用**：计算策略和价值函数的损失

```python
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators

loss_module = ClipPPOLoss(
    policy_module,
    value_module,
    clip_epsilon=0.2,         # PPO clip 参数 ε
    critic_coeff=0.5,         # 价值损失系数
    entropy_coeff=0.01,       # 熵正则化系数
    normalize_advantage=True  # 归一化优势函数
)

# 配置 GAE 优势估计器
loss_module.make_value_estimator(
    value_type=ValueEstimators.GAE,
    gamma=0.99,      # 折扣因子 γ
    lmbda=0.95       # GAE lambda λ
)
```

**数学原理**：

**PPO Clipped Objective**:

$$
L^\text{CLIP}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是 GAE 优势估计

**GAE (Generalized Advantage Estimation)**:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差

**总损失**:

$$
L_\text{total} = L^\text{CLIP} + c_1 L^\text{value} - c_2 H(\pi_\theta)
$$

- $L^\text{value} = \frac{1}{2}\mathbb{E}_t[(V_\phi(s_t) - V^\text{targ}_t)^2]$：价值损失
- $H(\pi_\theta) = -\mathbb{E}_t[\log \pi_\theta(a_t|s_t)]$：策略熵（鼓励探索）

---

### 2.5 Data Collector (数据收集器)

**作用**：与环境交互收集经验数据

```python
from torchrl.collectors import SyncDataCollector

collector = SyncDataCollector(
    torchrl_env,
    policy_module,
    frames_per_batch=16384,   # 每批收集的帧数
    total_frames=int(1e7),    # 总训练帧数
    device=device,
    max_frames_per_traj=env.max_steps  # 单条轨迹最大长度
)

# 迭代收集数据
for data in collector:
    # data 是 TensorDict，形状为 (num_envs, frames_per_batch//num_envs)
    print(data.keys())  # ["observation", "action", "reward", "done", "next_observation"]
```

**返回数据结构**：

```python
TensorDict({
    "policy": Tensor[num_envs, trajectory_length, policy_obs_dim],
    "critic": Tensor[num_envs, trajectory_length, critic_obs_dim],
    "action": Tensor[num_envs, trajectory_length, action_dim],
    "reward": Tensor[num_envs, trajectory_length],
    "done": Tensor[num_envs, trajectory_length],
    "next": TensorDict({...})  # 下一个状态的观测
})
```

---

## 3. PPO 算法实现剖析

让我们逐步分解 `train.py` 中的 PPO 训练循环：

### 3.1 数据收集阶段

```python
for data in collector:
    # data 形状: (num_envs, steps_per_env, ...)
    data = data.to(device)
    
    # 展平批次维度和时间维度
    # 从 (num_envs, steps) → (num_envs * steps,)
    rollout = data.flatten(0, 1)
```

**为什么要展平？**
- PPO 使用小批次 SGD 更新，需要打乱时间顺序
- 展平后可以随机采样 minibatch

---

### 3.2 优势函数计算

```python
# 使用 GAE 计算优势函数和目标价值
loss_module.value_estimator(rollout)
# 自动在 rollout 中添加:
#   - "advantage": GAE 优势估计
#   - "value_target": TD(λ) 目标价值
```

**内部实现**（简化版）：

```python
def compute_gae(rewards, values, dones, gamma=0.99, lmbda=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lmbda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

---

### 3.3 小批次训练

```python
for epoch in range(num_epochs):  # 多次遍历数据（默认 4 轮）
    for start in range(0, num_samples, minibatch_size):
        batch = rollout.narrow(0, start, minibatch_size)
        
        optimizer.zero_grad()
        losses = loss_module(batch)  # 计算损失字典
        total_loss = losses["loss_objective"] + losses["loss_critic"] - losses["loss_entropy"]
        total_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), grad_clip)
        optimizer.step()
```

**损失字典内容**：

```python
losses = {
    "loss_objective": -min(r*A, clip(r, 1-ε, 1+ε)*A),  # PPO clip 损失
    "loss_critic": 0.5 * (V - V_target)^2,             # 价值损失
    "loss_entropy": -H(π),                              # 熵损失（负值）
    "kl_approx": 0.5 * (log(π_new) - log(π_old))^2     # 近似 KL 散度
}
```

---

## 4. 非对称 Actor-Critic 实现

你的环境配置使用了非对称观测（Critic 有特权信息），这是 TorchRL 的一大优势。

### 4.1 观测结构

```python
# 环境配置 (inhand_base_env_cfg.py)
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):  # Actor 观测（本体感受）
        joint_pos = ObsTerm(...)
        last_action = ObsTerm(...)
        goal_pose = ObsTerm(...)
    
    @configclass
    class CriticCfg(ObsGroup):  # Critic 观测（特权信息）
        joint_pos = ObsTerm(...)
        object_pos = ObsTerm(...)      # ← 仅 Critic 可见
        object_quat = ObsTerm(...)     # ← 仅 Critic 可见
        goal_quat_diff = ObsTerm(...)  # ← 仅 Critic 可见
    
    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()
```

### 4.2 TorchRL 自动处理

```python
# observation_spec 结构
observation_spec = {
    "policy": BoundedTensorSpec(shape=(policy_obs_dim,)),
    "critic": BoundedTensorSpec(shape=(critic_obs_dim,))
}

# 构建 Actor（仅使用 policy 观测）
policy_module = build_actor(
    obs_key=("policy",),  # ← 键路径
    action_key=("action",),
    ...
)

# 构建 Critic（使用 critic 观测）
value_module = build_critic(
    obs_key=("critic",),  # ← 键路径
    ...
)
```

### 4.3 前向传播流程

```python
# 数据流动
tensordict = {
    "policy": policy_obs,   # 形状: (batch, policy_obs_dim)
    "critic": critic_obs    # 形状: (batch, critic_obs_dim)
}

# Actor 前向传播
policy_module(tensordict)
# 自动读取 tensordict["policy"]，写入 tensordict["action"]

# Critic 前向传播
value_module(tensordict)
# 自动读取 tensordict["critic"]，写入 tensordict["state_value"]
```

---

## 5. 与 RL-Games 的对比

| 维度 | RL-Games | TorchRL |
|------|----------|---------|
| **训练启动** | `runner.run({"train": True})` | 手动编写训练循环 |
| **算法切换** | 修改配置文件的 `algo` 字段 | 更换 Loss 模块和训练逻辑 |
| **自定义网络** | 继承 `BaseNet` 类 | 直接使用 PyTorch `nn.Module` |
| **非对称 AC** | 需要在 `BasePlayer` 中手动处理 | 原生支持嵌套 TensorDict |
| **调试难度** | 高（黑盒内部） | 低（每一步透明） |
| **扩展性** | 低（修改源码） | 高（组合组件） |
| **上手难度** | 简单 | 复杂 |

### 5.1 算法切换对比

**RL-Games**（修改配置文件）:
```yaml
params:
  algo:
    name: a2c_continuous  # 改成 sac 需要修改很多其他配置
```

**TorchRL**（更换损失模块）:
```python
# PPO → SAC 只需替换 Loss 和 Collector
# from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.sac import SACLoss

loss_module = SACLoss(
    actor_network=policy_module,
    qvalue_network=qvalue_module,
    ...
)

# 使用 Replay Buffer 替代同步收集器
from torchrl.data import ReplayBuffer
replay_buffer = ReplayBuffer(storage=..., sampler=...)
```

---

## 6. 自定义网络与算法

TorchRL 的最大优势是自定义能力。

### 6.1 自定义 Actor 网络

```python
class CustomActor(nn.Module):
    """使用 Transformer 的 Actor 网络"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.embed = nn.Linear(obs_dim, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        self.head = nn.Linear(256, 2 * action_dim)
    
    def forward(self, obs):
        x = self.embed(obs).unsqueeze(1)  # (batch, 1, 256)
        x = self.transformer(x).squeeze(1)
        loc, scale = self.head(x).chunk(2, dim=-1)
        return loc, torch.nn.functional.softplus(scale)

# 集成到 TorchRL
custom_actor = TensorDictModule(
    CustomActor(obs_dim, action_dim),
    in_keys=[("policy",)],
    out_keys=[("loc", "scale")]
)
policy_module = ProbabilisticActor(custom_actor, ...)
```

### 6.2 自定义训练循环（加入课程学习）

```python
for data in collector:
    rollout = data.flatten(0, 1)
    loss_module.value_estimator(rollout)
    
    # 自定义: 动态调整 clip_epsilon
    current_kl = compute_kl_divergence(rollout)
    if current_kl > target_kl * 1.5:
        loss_module.clip_epsilon *= 0.95  # 减小 clip 范围
    elif current_kl < target_kl * 0.5:
        loss_module.clip_epsilon *= 1.05  # 增大 clip 范围
    
    # 自定义: 课程学习（逐步增加任务难度）
    if global_frames > curriculum_threshold:
        env.set_difficulty(level=2)
    
    for epoch in range(num_epochs):
        # ... 正常训练
```

### 6.3 混合算法（PPO + Imitation Learning）

```python
# 在训练循环中同时使用 PPO 和行为克隆损失
ppo_loss_module = ClipPPOLoss(...)
bc_loss_module = BCLoss(...)  # 行为克隆

for data in collector:
    rollout = data.flatten(0, 1)
    
    # PPO 损失
    ppo_losses = ppo_loss_module(rollout)
    
    # 行为克隆损失（假设有专家轨迹）
    expert_batch = expert_buffer.sample()
    bc_loss = bc_loss_module(expert_batch)
    
    # 组合损失
    total_loss = (
        ppo_losses["loss_objective"] + 
        ppo_losses["loss_critic"] + 
        0.1 * bc_loss  # ← 混合系数
    )
    
    total_loss.backward()
    optimizer.step()
```

---

## 📚 **学习路径建议**

1. **基础阶段**：
   - 阅读 `train.py`，理解每个组件的作用
   - 运行训练并观察 TensorBoard 日志
   - 尝试修改网络结构（隐藏层大小、激活函数）

2. **进阶阶段**：
   - 实现自定义 Reward Shaping（在 `loss_module` 后手动调整 reward）
   - 添加自定义指标（如成功率、轨迹长度）到 TensorBoard
   - 尝试不同的优势估计器（TD(λ)、MC Return）

3. **高级阶段**：
   - 实现其他算法（SAC、TD3、DDPG）
   - 混合 On-Policy 和 Off-Policy 方法
   - 集成 Population-Based Training (PBT)

---

## 🎯 **快速参考**

### TorchRL 核心概念映射

| 概念 | TorchRL 类 | 作用 |
|------|-----------|------|
| 环境 | `TorchRLEnv` | 状态转移逻辑 |
| 策略 | `ProbabilisticActor` | 从观测生成动作 |
| 价值函数 | `ValueOperator` | 估计状态价值 |
| 损失计算 | `ClipPPOLoss` | 计算策略和价值损失 |
| 数据收集 | `SyncDataCollector` | 与环境交互收集经验 |
| 数据结构 | `TensorDict` | 嵌套张量字典 |
| 优势估计 | `GAE` | 计算优势函数 |

### 常见问题

**Q: 为什么需要 `TensorDictModule`？**  
A: 处理嵌套键（如 `("policy",)` 或 `("critic",)`），自动从 TensorDict 中读取/写入数据。

**Q: `flatten(0, 1)` 做了什么？**  
A: 将形状从 `(num_envs, steps, ...)` 变为 `(num_envs*steps, ...)`，便于小批次训练。

**Q: 如何调试 Loss 计算？**  
A: 在 `loss_module(batch)` 前后打印 `batch.keys()` 和 `losses` 字典。

---

## 🔗 **官方资源**

- [TorchRL 文档](https://pytorch.org/rl/)
- [TorchRL GitHub](https://github.com/pytorch/rl)
- [TensorDict 教程](https://pytorch.org/tensordict/stable/tutorials/)
- [PPO 论文](https://arxiv.org/abs/1707.06347)

---

**总结**：TorchRL 的组件化设计牺牲了易用性，但换来了极致的灵活性和透明度。对于研究者和需要高度自定义的项目，这是一个强大的工具。
