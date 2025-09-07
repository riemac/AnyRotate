---
type: "agent_requested"
description: "当每次对话开始时；当进行isaacLab项目开发，用来了解项目结构和开发规范"
---

# IsaacLab项目开发指南

## 项目说明
本文档用于记录IsaacLab项目开发过程中的关键信息和规范，便于团队成员快速了解和遵循项目结构。

## 工作区结构
```bash
leaphand/                    # 项目根目录
├── scripts/                 # 脚本目录
│   ├── list_envs.py        # 环境列表脚本
│   ├── random_agent.py     # 随机智能体示例
│   ├── zero_agent.py       # 零智能体示例
│   └── skrl/               # SKRL强化学习相关脚本
├── source/                  # 源代码目录
│   └── leaphand/           # 主要功能模块
│       ├── assets/         # 资源文件目录
│       │   └── *.usda      # 场景和模型文件
│       ├── docs/           # 项目文档
│       │   └── *.md        # Markdown格式文档
│       ├── leaphand/       # 核心功能实现
│       │   ├── robots/     # 机器人定义模块
│       │   │   └── leaphand.py
│       │   ├── tasks/       # 任务环境定义
│       │   │   ├── direct/  # DirectRLEnv
│       │   │   │   └── leaphand/
│       │   │   │       ├── leaphand_env.py # 任务环境实现
│       │   │   │       └── leaphand_env_cfg.py # 任务环境对应配置
│       │   │   └── manager_based/ # ManagerBasedRLEnv
│       │   │       └── leaphand/
│       └── pyproject.toml   # Python项目配置
├── logs/                    # 日志目录
└── outputs/                 # 输出目录

IsaacLab/                    # IsaacLab官方核心框架目录

LEAP_Hand_Isaac_Lab/           # LeapHand手官方的手内旋转项目（可参考）
```

## 开发规范

### 测试脚本
- 遵循standalone开发模式，使用appLauncher作为核心启动器
- 所有测试脚本统一放置在`scripts/`目录下
- 注意：某些依赖库需要IsaacSim环境启动后才能正常导入

### 文档管理
- 所有项目文档统一存放在`source/leaphand/docs/`目录
- 使用Markdown格式编写文档

## 注意项
- **导入路径:** 由于使用了uv pip install -e source/leaphand将Extension注册到python解析路径,导入相关环境和配置时，可按如下示例
```python
from leaphand.tasks.direct.leaphand.leaphand_env import LeaphandEnv
from leaphand.tasks.direct.leaphand.leaphand_env_cfg import LeaphandEnvCfg
```
- **环境激活:** 执行终端指令前一定要在 `~/isaac` 目录下激活 uv 环境: `source .venv/bin/activate`
