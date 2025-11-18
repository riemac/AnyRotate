#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录
cd /home/hac/isaac/AnyRotate || { echo "cd /home/hac/isaac/AnyRotate 失败"; exit 1; }

# 激活 Isaac 环境（优先 env_isaac，兼容 env_isaaclab）
if [[ -f ~/isaac/env_isaac/bin/activate ]]; then
    source ~/isaac/env_isaac/bin/activate
elif [[ -f ~/isaac/env_isaaclab/bin/activate ]]; then
    source ~/isaac/env_isaaclab/bin/activate
else
    echo "找不到 Isaac 环境激活脚本（~/isaac/env_isaac/bin/activate 或 env_isaaclab）"
    exit 1
fi

# 默认参数（可通过环境变量或传参覆盖）
AMPLITUDE="${AMPLITUDE:-0.3}"
FREQUENCY="${FREQUENCY:-1.0}"
STIFFNESS="${STIFFNESS:-5.0}"
DAMPING="${DAMPING:-0.5}"
DURATION="${DURATION:-10.0}"
DECIMATION="${DECIMATION:-4}"

# 运行 IsaacLab，$@ 允许在调用脚本时额外传参覆盖
exec ./IsaacLab/isaaclab.sh -p source/leaphand/leaphand/tasks/manager_based/test_actuator.py \
    --amplitude "$AMPLITUDE" \
    --frequency "$FREQUENCY" \
    --stiffness "$STIFFNESS" \
    --damping "$DAMPING" \
    --duration "$DURATION" \
    --decimation "$DECIMATION" "$@"
