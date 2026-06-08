# 离线强化学习项目计划文档

## 1. 项目目标

本项目的目标是系统复现并理解三类具有代表性的离线强化学习算法：

* IQL：Implicit Q-Learning
* CQL：Conservative Q-Learning
* DT：Decision Transformer

在完成可靠 baseline 复现之后，我们将基于对数据分布和算法机制的理解，提出一个小而明确的改进方向，并在 D4RL benchmark 上进行验证。

本项目不只是为了跑出分数，而是为了理解：

* 离线强化学习数据集到底包含什么；
* 为什么 Offline RL 比 Online RL 更困难；
* 不同算法如何处理 distribution shift 和 OOD action；
* 数据质量如何影响算法性能；
* 是否可以基于数据分布或算法机制提出一个可控的小改进。

---

## 2. 项目定位

本项目的核心定位是：

> 先系统复现 Offline RL 三类代表方法：IQL、CQL、Decision Transformer；再基于对数据分布与算法机制的理解，提出一个小而明确的改进方向，并在 D4RL benchmark 上验证。

项目优先级如下：

1. 可复现性；
2. 实验管理清晰；
3. 算法理解深入；
4. 数据集分析充分；
5. 改进方向小而明确；
6. 结果解释有逻辑。

---

## 3. 统一代码库选择

baseline 阶段统一使用 CORL 作为主代码库。

仓库地址：

```text
https://github.com/tinkoff-ai/CORL
```

本地路径：

```text
/code/CORL
```

选择 CORL 的原因：

* CORL 提供了较干净的 PyTorch 单文件实现；
* 已经支持 IQL、CQL 和 Decision Transformer；
* D4RL evaluation protocol 较统一；
* 便于两人协作和对齐实验结果；
* 避免不同实现之间因细节差异造成结果不可比。

重要原则：

> baseline 实验必须统一基于 CORL。
> 从零实现的代码可以用于学习、理解或 sanity check，但不作为主 baseline，除非双方明确同意。

---

## 4. 两人分工

### 成员 A：算法负责人

主要职责：

* 阅读 IQL、CQL、Decision Transformer 论文；
* 阅读 CORL 中对应算法实现；
* 建立论文公式与代码实现之间的对应关系；
* 理解每个算法的 loss、网络结构、训练流程和关键超参数；
* 分析不同数据集分布对算法的影响；
* 提出后续改进方向；
* 撰写算法理解报告和最终方法报告。

主要交付物：

```text
/code/reports/IQL_CODE_REPORT.md
/code/reports/CQL_CODE_REPORT.md
/code/reports/DT_CODE_REPORT.md
/code/reports/D4RL_DATASET_REPORT.md
/code/reports/METHOD_PROPOSAL.md
/code/reports/FINAL_REPORT.md
```

---

### 成员 B：实验负责人

主要职责：

* 维护 CORL 实验环境；
* 运行 IQL、CQL、DT baseline；
* 管理日志、checkpoint、训练脚本和结果表格；
* 保证每个实验可复现、可追踪；
* 监控失败实验并记录原因；
* 汇总不同算法、数据集、随机种子下的结果；
* 绘制 score 曲线和 loss 曲线。

主要交付物：

```text
/code/scripts/
/code/logs/
/code/checkpoints/
/code/results/baseline_results.csv
/code/results/experiment_status.csv
/code/reports/BASELINE_RESULTS.md
```

---

## 5. 推荐项目目录结构

建议在 `/code` 下统一组织项目文件：

```text
/code/CORL
/code/scripts
/code/logs
/code/checkpoints
/code/datasets
/code/results
/code/reports
/code/analysis
```

各目录含义：

```text
/code/CORL         # 主代码库
/code/scripts      # 可复现的运行脚本
/code/logs         # stdout/stderr 日志
/code/checkpoints  # 模型 checkpoint
/code/datasets     # 持久化 D4RL 数据集
/code/results      # CSV 结果表格
/code/reports      # 人类可读的项目报告
/code/analysis     # 数据分析和代码分析脚本
```

注意：

> 尽量不要把重要文件只存放在 `/root` 下。
> 数据集、日志、checkpoint 和结果表格应尽量放在 `/code` 这样的持久化目录中。

---

## 6. Baseline 算法

第一阶段复现以下三个算法：

```text
IQL
CQL
Decision Transformer
```

它们分别代表三类 Offline RL 思路：

| 算法  | 类型                              | 核心思想                                                                           |
| --- | ------------------------------- | ------------------------------------------------------------------------------ |
| IQL | Value-based / policy extraction | 通过 expectile value learning 和 advantage-weighted BC 避免 OOD action maximization |
| CQL | Conservative value learning     | 压低 OOD action 的 Q 值，缓解 Q overestimation                                        |
| DT  | Sequence modeling               | 将 RL 看成条件轨迹建模问题                                                                |

---

## 7. 初始实验矩阵

### 7.1 最小 baseline

首先完成：

```text
算法：
- IQL
- CQL
- DT

数据集：
- halfcheetah-medium-v2
- hopper-medium-v2
- walker2d-medium-v2

随机种子：
- seed 0
```

总实验数：

```text
3 algorithms × 3 datasets × 1 seed = 9 runs
```

这一步的目标是验证：

* 环境是否稳定；
* CORL 中三个算法是否都能跑通；
* 日志和结果记录是否规范；
* baseline 分数是否处在合理范围。

---

### 7.2 正式 baseline

在最小 baseline 稳定之后，扩展为：

```text
算法：
- IQL
- CQL
- DT

数据集：
- halfcheetah-medium-v2
- hopper-medium-v2
- walker2d-medium-v2

随机种子：
- seed 0
- seed 1
- seed 2
```

总实验数：

```text
3 algorithms × 3 datasets × 3 seeds = 27 runs
```

这一步将形成项目最核心的 baseline 结果表。

---

### 7.3 扩展数据集

如果时间允许，进一步加入：

```text
halfcheetah-medium-replay-v2
hopper-medium-replay-v2
walker2d-medium-replay-v2

halfcheetah-medium-expert-v2
hopper-medium-expert-v2
walker2d-medium-expert-v2
```

这些数据集更适合观察 Offline RL 中的数据分布问题：

* `medium-replay`：训练过程 replay buffer，数据分布更混乱；
* `medium-expert`：medium 与 expert 混合，数据具有多模态特征；
* `expert`：质量高但 support 较窄。

---

## 8. 实验命名规范

每个实验命名建议采用：

```text
{algorithm}_{env}_{seed}
```

例如：

```text
iql_halfcheetah_medium_seed0
cql_hopper_medium_seed1
dt_walker2d_medium_seed2
```

日志路径：

```text
/code/logs/{algorithm}/{env}/seed_{seed}.log
```

checkpoint 路径：

```text
/code/checkpoints/{algorithm}/{env}/seed_{seed}/
```

结果表中每一行应至少包含：

```text
algorithm, env, dataset_type, seed, status, final_score, best_score, runtime, log_path, checkpoint_path
```

---

## 9. 实验结果表

维护统一结果文件：

```text
/code/results/baseline_results.csv
```

推荐字段：

```text
algorithm
env
dataset_type
seed
status
final_d4rl_score
best_d4rl_score
runtime_hours
log_path
checkpoint_path
notes
```

示例：

```text
IQL, halfcheetah-medium-v2, medium, 0, completed, 47.3, 48.1, 4.5, /code/logs/iql/halfcheetah-medium-v2/seed_0.log, /code/checkpoints/iql/halfcheetah-medium-v2/seed_0, ok
```

---

## 10. 当前第一里程碑：IQL

优先从 IQL 开始，因为：

* IQL 已经在当前环境中跑通；
* 算法思想清晰；
* 代码实现相对简洁；
* 很适合作为 Offline RL 的第一篇深入理解对象。

当前 IQL 任务：

1. 完成 `halfcheetah-medium-v2` 的完整 IQL 训练；
2. 运行：

   * `hopper-medium-v2`
   * `walker2d-medium-v2`
3. 阅读并解释 CORL 中：

   * `TrainConfig`
   * `ReplayBuffer`
   * `GaussianPolicy`
   * `DeterministicPolicy`
   * `TwinQ`
   * `ValueFunction`
   * `ImplicitQLearning`
   * `train()`
4. 生成：

   * `/code/reports/IQL_CODE_REPORT.md`

---

## 11. 算法理解报告规范

对每个算法，算法负责人应产出：

```text
/code/reports/{ALGORITHM}_CODE_REPORT.md
```

每份报告应包含：

1. 算法要解决的 Offline RL 核心问题；
2. 该算法如何处理 OOD action / distribution shift；
3. 主要数学目标；
4. 代码结构；
5. 论文公式与代码实现对应关系；
6. 关键超参数；
7. 训练动态与 loss 曲线解释；
8. 数据集分布对算法表现的影响；
9. 算法优点；
10. 算法局限；
11. 可能的改进点。

---

## 12. 数据集理解报告

创建：

```text
/code/reports/D4RL_DATASET_REPORT.md
```

数据集分析至少包括：

* observations；
* actions；
* rewards；
* next_observations；
* terminals；
* trajectory length distribution；
* trajectory return distribution；
* action distribution；
* state distribution；
* reward histogram；
* state/action PCA；
* dataset quality；
* dataset coverage；
* OOD action risk。

初始分析数据集：

```text
halfcheetah-medium-v2
hopper-medium-v2
walker2d-medium-v2
```

后续扩展：

```text
medium-replay
medium-expert
```

该报告的重点不是简单统计，而是从 Offline RL 角度解释：

* 数据分布是否窄；
* 是否存在多模态；
* behavior policy 是否稳定；
* 哪些地方可能产生 OOD action；
* IQL / CQL / DT 分别会如何受数据分布影响。

---

## 13. 候选改进方向

暂时不追求“大而全”的新算法，优先考虑小而明确、容易验证的改进。

### 方向 A：改进 IQL 的 advantage weighting

原始 IQL actor 权重：

```text
weight = exp(beta * advantage)
```

可能问题：

* advantage 尺度在不同数据集上差异较大；
* 固定 beta 可能不稳定；
* 极端 advantage 可能导致少数样本权重过大。

可能改进：

* normalized advantage；
* clipped advantage；
* rank-based advantage weight；
* beta schedule；
* soft clipping。

优点：

* 实现较小；
* 与 IQL 核心机制直接相关；
* 容易做 ablation；
* 结果容易解释。

---

### 方向 B：return-aware 或 advantage-aware sampling

当前 IQL/CQL 通常均匀采样 transition。

可以尝试根据以下信息重加权采样：

* trajectory return；
* reward；
* estimated advantage；
* uncertainty；
* trajectory quality。

研究动机：

* Offline RL 很依赖数据分布；
* 低质量数据可能拖累策略；
* 但过度过滤低质量数据又会降低 coverage。

---

### 方向 C：数据集自适应的 IQL 超参数

根据数据集统计动态调整：

```text
beta
iql_tau
```

可能依据：

* reward scale；
* advantage distribution；
* action distribution；
* trajectory return distribution；
* dataset quality。

---

### 方向 D：IQL 与 DT 的结合

利用 IQL 学到的 value / advantage 信息，对 DT 的 trajectory 或 action token 进行加权。

该方向更有野心，但工程复杂度更高。建议在完成稳定 baseline 之后再考虑。

---

## 14. 时间安排建议

### 第 1 周：IQL 与数据理解

目标：

```text
IQL 在 halfcheetah / hopper / walker2d medium 上跑通
完成 IQL_CODE_REPORT.md
完成 D4RL_DATASET_REPORT.md 初版
```

交付物：

```text
IQL 实验结果
/code/reports/IQL_CODE_REPORT.md
/code/reports/D4RL_DATASET_REPORT.md
/code/results/iql_results.csv
```

---

### 第 2 周：CQL 与 DT baseline

目标：

```text
CQL 跑通
DT 跑通
三个算法在 medium 数据集上形成初步对比
```

交付物：

```text
/code/reports/CQL_CODE_REPORT.md
/code/reports/DT_CODE_REPORT.md
/code/results/baseline_results.csv
/code/reports/BASELINE_RESULTS.md
```

---

### 第 3 周：多 seed 与扩展数据集

目标：

```text
medium 数据集上完成 3 seeds
初步加入 medium-replay / medium-expert
分析不同数据分布下算法表现
```

交付物：

```text
/code/results/full_baseline_table.csv
/code/reports/DATASET_ALGORITHM_ANALYSIS.md
score curves
loss curves
```

---

### 第 4 周：小改进与 ablation

目标：

```text
选择一个改进点
实现修改
完成 ablation
撰写总结
```

交付物：

```text
/code/reports/METHOD_PROPOSAL.md
改进版算法实现
/code/results/ablation_results.csv
/code/reports/FINAL_REPORT.md
```

---

## 15. 协作规则

1. baseline 代码应尽量保持 CORL 原始逻辑；
2. 算法修改必须放在单独文件或单独分支中；
3. 不要覆盖已有日志和 checkpoint；
4. 每个实验必须有唯一名字；
5. 每个结果必须能追溯到：

   * 代码版本；
   * 启动命令；
   * 环境名；
   * seed；
   * 日志文件；
   * checkpoint；
6. 算法负责人和实验负责人应在每批实验结束后同步；
7. 失败实验也要记录，不要直接删除；
8. 实验结果不能只看最终分数，还要看训练曲线和稳定性；
9. 如果两人使用不同 AI coding 工具，也必须统一代码库和结果格式；
10. 从零生成的代码可以用于理解，但 baseline 以 CORL 为准。

---

## 16. 当前立即行动项

### 算法负责人

当前任务：

* 继续阅读 CORL 的 IQL 实现；
* 完成以下部分的理解：

  * `ReplayBuffer`
  * `GaussianPolicy`
  * `TwinQ`
  * `ValueFunction`
  * `ImplicitQLearning`
  * `train()`
* 起草 `/code/reports/IQL_CODE_REPORT.md`；
* 梳理 IQL 的公式与代码映射；
* 分析 IQL 在 `halfcheetah-medium-v2` 上的训练结果。

---

### 实验负责人

当前任务：

* 继续当前 IQL 完整训练；
* 启动 IQL on：

  * `hopper-medium-v2`
  * `walker2d-medium-v2`
* 统一保存日志和 checkpoint；
* 创建并维护：

  * `/code/results/baseline_results.csv`
  * `/code/results/experiment_status.csv`
* 确认 D4RL 数据集持久化缓存；
* 后续准备 CQL 和 DT 的运行脚本。

---

## 17. 当前阶段原则

项目当前阶段不应停留在：

```text
代码能不能跑？
```

而应转向：

```text
我们是否理解数据、算法和结果？
```

之后再进入：

```text
我们能否提出一个有根据的小改进？
```

---

## 18. 给实验负责人的说明

如果之前已经使用 Claude Code 从零生成了一些实验代码，可以保留作为辅助理解或 sanity check。

但正式 baseline 应迁移到 CORL，原因是：

* 两人更容易对齐；
* 结果更容易比较；
* 实验流程更标准；
* IQL / CQL / DT 在同一代码库中实现；
* 避免不同实现带来的不可控差异。

统一原则：

> Claude Code / Cursor 可以作为执行工具。
> CORL 是 baseline 的主代码库。
> 所有正式结果进入统一结果表。
