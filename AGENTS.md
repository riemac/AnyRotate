# AGENTS.md

## Project Overview

基于Isaac Lab框架的机器人仿真环境开发规范文档。本项目专注于LeapHand机械手的强化学习训练环境构建。

## 核心原则

1. **🤖 主动确认，持续循环 (Proactive Confirmation, Continuous Loop)**

   * **操作机制:**

     1. **输出方案:** 在对话中输出完整的下一步方案（代码、计划、分析等）。
     2. **请求确认:** 调用 `mcp-feedback-enhanced` 工具，用一句话摘要请求批准。
     3. **执行:** 获得批准后执行该步骤。
   * **循环机制:**
     一旦步骤执行完毕，必须再次调用 `mcp-feedback-enhanced`，准备进入下一步。
   * **终止条件:** 仅当收到“任务结束”、“停止对话”等明确指令时循环终止。

2. **🧐 事实驱动，杜绝猜测**

   * **信息源:** 不分先后，具情况选用

     1. 本地代码（IsaacLab 源码、项目文档、示例，使用`codebase`检索工具）。
     2. 官方文档（`context7` 工具，当查询isaacsim, physx, torchrl, rl_games等第三方库）。
     3. 网络搜索（`github`, `fetch`工具）。

## 工作区结构

```bash
leaphand/                    # 个人Leaphand项目根目录
├── scripts/                 # 脚本目录
│   ├── debug/              # 调试脚本目录
│   ├── demo/               # 演示脚本目录
│   ├── evaluate/           # 评估脚本目录
│   ├── list_envs.py        # 环境列表脚本
│   ├── random_agent.py     # 随机智能体示例
│   ├── zero_agent.py       # 零智能体示例
│   └── rl_games/           # rl_games 强化学习相关脚本
├── source/                  # 源代码目录
│   └── leaphand/           # 主要功能模块
│       ├── assets/         # 资源文件目录 (*.usda)
│       ├── docs/           # 项目文档 (*.md)
│       ├── leaphand/       # 核心功能实现
│       │   ├── robots/     # 机器人定义
│       │   │   └── leap.py
│       │   ├── tasks/      # 任务环境定义
│       │   │   ├── direct/leaphand/   # DirectRLEnv
│       │   │   └── manager_based/leaphand/   # ManagerBasedRLEnv（目前主要开发目录）
│       └── pyproject.toml   # Python 项目配置
├── logs/                    # 日志目录
└── outputs/                 # 输出目录

IsaacLab/                    # IsaacLab官方核心框架目录（位于~/isaac/IsaacLab）

LEAP_Hand_Isaac_Lab/         # LeapHand官方手内重定向项目（供参考）（位于~/isaac/LEAP_Hand_Isaac_Lab）

LEAP_Hand_Sim/               # 早期基于isaacgym的LeapHand官方手内旋转项目（可参考）（位于~/LEAP_Hand_Sim）
```

## 工作流程

| 流程        | 步骤                                                                                                               |
| :-------- | :--------------------------------------------------------------------------------------------------------------- |
| **新功能开发** | 1. **规划:** 在对话中提出实现计划。<br>2. **确认:** 调用 `mcp-feedback-enhanced` 请求批准。<br>3. **执行:** 获得同意后，编码实现。                  |
| **错误调试**  | 1. **收集:** 在终端中定位关键错误信息。<br>2. **分析:** 在对话中结合日志提出原因假设和调试步骤。<br>3. **确认:** 调用 `mcp-feedback-enhanced` 请求批准后，开始调试。 |


## 开发规范

### 脚本开发

* 脚本开发通常是为了验证、调试或评估特定功能。
* Isaaclab项目提供了两个基础脚本：`list_envs.py` 用于列出所有可用环境，`random_agent.py` 该用于验证环境正确性和集成功能。
* 注意脚本复用，一般`random_agent.py`可测试大部分内容，若不满足的情况才需要开发新的脚本。
* 如若开发脚本，应按照性质放在`scripts/`目录下对应的子目录中，`debug/`用于调试功能，`demo/`用于演示功能（含可视化），`evaluate/`用于评估功能。
* 遵循 standalone 开发模式，使用 **appLauncher** 作为核心启动器。
* 注意：部分依赖库需在 **IsaacSim 环境启动后** 才能正常导入。

### 文档管理

* 所有项目文档统一存放在 `source/leaphand/docs/` 目录。

## 注意事项

### 操作要求

* **环境激活:**
  执行终端指令前，必须在 `~/isaac` 目录下激活 uv 环境：

  ```bash
  source .venv/bin/activate
  ```

* **反馈增强:**
  `mcp-feedback-enhanced` 遇到超时/失败时，必须再次调用。

### 工程问题

* **环境原点偏置:**
  多环境并行训练时，每个环境实例有自己的原点偏置 (`env_origins`)。开发时需要考虑这个偏置对位置、姿态等计算的影响。

* **坐标表示与转换:**
  注意使用的是World坐标系还是Body坐标系，以及如何正确转换。

* **任务环境导入路径:**
  项目使用 `uv pip install -e source/leaphand` 将 Extension 注册到 Python 解析路径，可直接导入：
  ```python
  from leaphand.tasks.direct.leaphand.leaphand_env import LeaphandEnv
  from leaphand.tasks.direct.leaphand.leaphand_env_cfg import LeaphandEnvCfg
  ```
* **环境步数:**
  ManagerBasedRLEnv 的 `common_step_counter` 是针对所有环境的共同步数，不是单独环境步数×环境数。在课程学习中需注意区分。

### 个人偏好

* **数理回复:**
  解释算法等机理性内容，结合数学公式。简洁美观的经渲染数学公式比大段文字和代码描述更易懂。

* **注释风格:**
  实现复杂方法时，在``` ```字符串中使用[NOTE:数学公式]来描述算法。如下所示
    ```python
    """计算旋转速度奖励 - 目标是达到指定的角速度而非越快越好

    Args:
        env: ManagerBasedRLEnv - 环境实例
        asset_cfg: SceneEntityCfg - 物体资产配置
        visualize_actual_axis: bool - 是否可视化实际旋转轴
        target_angular_speed: float - 目标角速度大小 (rad/s)
        positive_decay: float - 正向奖励的指数衰减因子
        negative_penalty_weight: float - 负向惩罚的权重系数

    Returns:
        旋转速度奖励 (num_envs,)

    Note:
        旋转轴是绕的世界坐标系中的固定轴旋转，而不是绕物体自身的局部坐标系轴旋转
        物体旋转时的旋转轴和Body Frame的表示无关
        奖励公式：
        - 正向速度: R = exp(-positive_decay * |projected_velocity - target_angular_speed|)
        - 负向速度: R = negative_penalty_weight * projected_velocity (负惩罚)
    """
    ```

* **表格对比:**
  总结涉及到众多复杂可比项或时，使用表格进行对比。

## 代码实践

* **代码隔离:** 绝不修改 IsaacLab 核心代码，开发在独立项目中进行。
* **风格一致:** 代码与项目风格与 IsaacLab 保持一致。
* **善用框架:** 优先利用 IsaacLab 现有功能，避免重复造轮子。


