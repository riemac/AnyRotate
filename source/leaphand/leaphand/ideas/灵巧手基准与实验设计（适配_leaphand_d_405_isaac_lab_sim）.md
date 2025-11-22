# 灵巧手基准与实验设计（适配 LEAPHand + D405 + IsaacLab/Sim）

**短摘要**

本文档把代表性的灵巧手（dexterous hand）基准任务（in‑hand / articulated / long‑horizon）与常见评测协议整理成一个可直接拷贝到论文 Methods/Experiments 的模板，并在末尾结合你已有设备（LEAPHand、Intel D405 深度相机、无触觉传感器）与仿真平台（IsaacLab/IsaacSim）给出**推荐实验组与具体实现建议**。

---

## 1. 目标与设计原则（为什么这套基准）
- 覆盖不同难度与能力维度：**精细 in‑hand 控制 → 铰接/接触操作 → 长时序/多步骤操作**。
- 评估指标兼顾**成功率、效率（steps/time）、sample efficiency、鲁棒性、泛化性**。
- 在论文中需要能复现/对比的细节（种子数、observation/action 定义、reward/success 判定、baseline 算法）。

---

## 2. 推荐基准任务（简要，便于在论文里引用与实现）
- **In‑hand 精细控制（示例）**
  - 任务描述：例如 pen re‑orientation / in‑hand rotation、小物体定向放置。
  - 测评要点：末端物体位姿误差、成功率、平均完成步数。

- **铰接/关节对象操作（示例）**
  - 任务描述：旋转阀门、开关、抽屉操作等，需要稳定接触与力矩控制。
  - 测评要点：成功率、对未见对象（物理参数变化）的泛化能力。

- **长时序/组合任务（示例）**
  - 任务描述：多次 regrasp、组合搬运或 Rubik’s‑like 操作序列。
  - 测评要点：长期成功率、累积错误率、策略稳定性。

---

## 3. 通用实验协议模板（论文中可直接粘贴）
- **随机种子与重复次数**：每个实验至少运行 5 个不同随机种子；对关键结果报告均值 ± 标准差。
- **评价指标**：Success (%), Avg steps to success, Final pose error (cm / deg), Sample efficiency 曲线（reward vs env steps）、鲁棒性曲线（对扰动/摩擦/质量的失败率）。
- **观测与动作（默认）**：
  - Observation = [关节位置+速度; 相机 RGBD（或 depth）投影的目标物体局部视图; （可选）末端触点力/力矩（若仿真可获得）]
  - Action = [关节目标位置或关节速度/力矩指令]
- **训练策略 / baselines**：PPO、SAC、SAC+HER（sparse reward 时）、BC / DAPG（若有示范）、Diffusion policy / DT 风格的行为克隆（如需比样本效率）。
- **成功判定**：明确且自动判定（例如物体姿态误差 < x deg 且位移 < y cm 且持续 T steps）。

---

## 4. 针对你现有设备的约束与映射（LEAPHand、D405、无触觉、IsaacLab/Sim）
- **LEAPHand**：作为手硬件/模型，使用其关节自由度（DOF）、运动学与控制接口（位置/速度/力控）作为 action space。仿真中建议尽量复现真实手的摩擦、关节阻尼与舵机延迟。

- **D405 相机（RGB + Depth）**：
  - 优先使用 RGBD（或 Depth + Mask）作为主要感知输入，替代触觉传感器用于感知接触/物体姿态。D405 在近距深度感知与高分辨率上有优势；实际搭建时注意相机安装视角与仿真视角的一致性（内参/外参）。
  - 在仿真中使用 IsaacSim 的深度相机/相机 sensor 模拟 D405 的输出（同分辨率与视场，仿真加入噪声）。

- **无触觉传感器的限制与补救策略**：
  - 直接实物实验无法衡量真实触觉，但在仿真里可以开启 contact forces / local pressure proxies 作为 "仿真触觉" 用于 ablation。把仿真触觉作为可选输入，这样能比较“vision‑only vs vision+simulated tactile”。
  - 在没有物理触觉硬件时，推荐利用视觉和物理推断（例如深度图 + 物体碰撞几何/penetration depth）做接触判定与局部策略调整。

- **IsaacLab / IsaacSim 优势**：
  - 能模拟高质量的 RGB/D 数据、物理接触、以及多传感器融合，方便做 sim→real（用 ADR/域随机化）。
  - 建议在仿真中复现 LEAPHand 的关节限制、传感器安装和相机参数，确保 sim/real 视角和观测分布尽量一致。

---

## 5. 每个任务的具体实现建议（伪码级别思路，便于在 IsaacLab/ManiSkill 中实现）
> **注**：下面为高层伪码思路（用于论文 Methods）。实现时需把参数替换为你自己的 env/hand model 的具体数值。

### 5.1 任务 A — In‑hand reorientation（适配 LEAPHand）
- **Obs**：LEAPHand 关节角/速度 + 相机 RGBD 局部 crop（物体 tight bounding box） + 目标姿态（task condition）。
- **Action**：关节位置目标（delta position）或速度命令。
- **Reward**：-α * pose_error_deg − β * transl_error_cm − γ * sum(|action|)；稀疏条件可在 pose_error < thresh 给 +1。
- **Success 判定**：物体姿态误差 < 10° 且位置误差 < 2 cm，且维持 10 steps。
- **实现提示（IsaacSim）**：在仿真中用 depth + segmentation mask 获取物体局部裁剪；在训练时对 depth 加高斯噪声与随机遮挡。

### 5.2 任务 B — 铰接对象（旋转阀门 / 抽屉）
- **Obs**：关节角/速度 + 相机 RGBD（观察阀门手柄）+ 当前阀门角度（若可估计）。
- **Action**：同上。
- **Reward**：以阀门目标角度与当前角度差值的负值为主，加上接触稳定性奖励。
- **泛化设计**：训练集使用若干阀门几何/摩擦参数，测试集使用未见的几何与摩擦组合。

### 5.3 任务 C — 长时序 regrasp / multi‑step
- **Obs**：全程关节/视觉观测，分段目标（subgoal scheduling）。
- **Action**：高层动作（grasp primitive + regrasp primitive）或低层关节命令。建议试验 hierarchical RL（高层策划子任务，低层完成）。
- **Reward**：任务完成奖励 + 中间阶段达成奖励。

---

## 6. 推荐 baseline 和 ablation（论文里应包含）
- **Baselines**：PPO / SAC / SAC+HER（若 sparse）/ BC 或 DAPG（若示范可用）/（可选）Diffusion policy。
- **Ablations**：
  1. Vision‑only（RGBD + proprio） vs Vision+simulated tactile（仿真 contact forces）。
  2. With / Without ADR（域随机化）— 看 sim→real 成本。
  3. Action type ablation：Position target vs Velocity vs Torque。

---

## 7. Practical sim→real 流程建议（针对 LEAPHand + D405）
1. **相机对齐**：在真实平台把 D405 固定到训练中所用的视角，高精度测量并把相机内参/外参导入仿真。确保图像分辨率、FOV 与仿真匹配。
2. **观测预处理一致性**：在仿真中加入与真实相机相近的噪声（深度系数误差、随机遮挡、光照变化），同时实现相同的图像预处理管线（crop/resize/normalization）。
3. **域随机化（ADR）**：在训练时对物体材质、光照、摩擦、质量、关节延迟进行随机化，并记录哪些随机化是对真实转移最有效的。
4. **少量真实微调**：若可能，用少量真实试验（同任务示范或 RL fine‑tune）来收敛策略。
5. **评估**：在真实环境下用 D405 提供的 RGB+D 观测运行策略，并记录成功率、失败模式与视频证据。

---

## 8. 推荐的论文里可直接复制的实验组（针对你的硬件）
下面给出**三组主实验**，每组包含控制变量、评测指标与对比项，方便直接放进论文 Experiments：

### 实验组 1 — In‑hand 精细控制（仿真评估）
- **目标**：证明基方法在精细物体操控能力（in‑hand）上的表现与 sample efficiency。
- **平台**：IsaacSim（复现 LEAPHand）
- **观测**：LEAPHand proprio + 相机 RGBD（局部 crop）
- **动作**：关节位置目标（delta）
- **训练算法**：SAC（主）; 比较 PPO、SAC+HER（若使用稀疏奖励）、BC（有示范时）
- **指标**：Success (%), Avg steps, Reward vs training steps 曲线。
- **Ablation**：Vision‑only vs Vision+simulated tactile。

### 实验组 2 — 铰接对象泛化（train/test 未见对象）
- **目标**：评估方法对未见铰接对象（阀门/开关/抽屉）的泛化能力。
- **平台**：IsaacSim（训练集/测试集分开）
- **观测**：LEAPHand proprio + 全局/局部 RGBD
- **训练算法**：PPO 或 SAC，尝试 domain randomization（ADR）设置
- **指标**：Train success on seen objects, Test success on unseen objects（泛化 gap），Robustness to friction/pose perturbation
- **Ablation**：有/无 ADR；有/无仿真触觉

### 实验组 3 — 长时序 multi‑step（仿真 → 部分真实验证）
- **目标**：测试长序列任务（如 regrasp 或 Rubik’s‑like）中策略稳定性与 sim→real 能力。
- **平台**：IsaacSim 训练；若条件允许，把最有希望的策略部署到真实 LEAPHand + D405 上做有限次（例如 20 次）实测。
- **观测/动作**：同上，建议尝试 hierarchical policy（高层子任务 + 低层轨迹控制）
- **指标**：Cumulative success rate, Steps to first failure, Real world transfer success (%)
- **Ablation**：有/无 ADR；高层是否使用视觉子目标。

---

## 9. 实验记录与可复现性建议（便于论文附录）
- 给出 hyper‑params 表格（学习率、batch size、gamma、网络结构、training steps）。
- 报告随机种子、训练时长（env steps）、训练时的 compute 资源（GPU/CPU 型号）。
- 附录放置 reward 公式、success 判定准则、相机内参与安装位置图片、以及简短代码片段说明如何采集 D405 数据流。

---

## 10. 可选扩展（后续工作）
- 考虑加入外部传感器（触觉或力/触点传感器）作为对比；或在仿真中生成大规模 contact signal dataset 供后续研究者使用。
- 使用 imitation + RL 混合（例如先 BC 预训练，再 RL fine‑tune）以提高 sample efficiency。

---


**结束语 — 立即可用的行动项（简短）**
1. 在 IsaacLab 中建立 LEAPHand 的仿真 URDF/usd，配置 D405 相机视角并输出 RGBD 流。
2. 从实验组 1 开始（in‑hand reorientation），用 SAC 训练一个 baseline 并记录成功率曲线；同时做 vision‑only 与 vision+simulated‑tactile 的 ablation。
3. 在完成仿真评估且取得可重复效果后，再逐步推进实验组 2 与 3，并尝试使用 ADR 做 sim→real 转移。



---

*若你希望我把上面某一实验组翻译成可直接运行的伪代码、训练超参表或用于论文的 Methods 段落（英文），我可以直接生成。*

