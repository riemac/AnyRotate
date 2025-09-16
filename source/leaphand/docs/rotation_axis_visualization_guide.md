# LeapHand连续旋转环境 - 旋转轴可视化指南

## 功能与定义

- 旋转轴在世界坐标系中定义为固定方向向量
- 箭头始终指向该固定方向，位置跟随物体移动以便观察
- 遵循右手螺旋定则

## 启用

```python
# 在环境配置中启用
commands.rotation_axis.debug_vis = True
```

或：
```python
rotation_axis = RotationAxisCommandCfg(
    # ...其它配置...
    debug_vis=True,
)
```

## 运行

```bash
cd ~/isaac && source .venv/bin/activate
cd leaphand
python scripts/rl_games/train.py --task=Isaac-Leaphand-ContinuousRot-Manager-v0 --num_envs=4
```

## 参数

```python
visualizer_cfg = RotationAxisVisualizerCfg(
    enabled=True,
    offset_above_object=0.15,
    arrow_length=0.12,
    arrow_thickness=0.008,
    arrow_color=(1.0, 0.2, 0.2),
    opacity=0.8,
)
```

## 故障排除

- 确认 debug_vis=True
- 检查场景是否正确加载
- 调整 offset/length/thickness 以获得更清晰显示
