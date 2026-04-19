# llm_learn 下一阶段补丁报告（Qwen 评测入口 + 示例数据模板 + 1.5B/7B 配置）

## 这次补丁做了什么

这次补丁是在上一阶段 `src/qwen/` 最小训练骨架的基础上，继续把项目往“可复用的 Qwen 微调仓库”推进，主要新增三块能力：

1. **评测入口**：补上 `qwen-sft evaluate` 命令，能读取评测集、生成预测、输出 `predictions.jsonl` 和 `metrics.json`。
2. **示例数据模板**：补上 `data/qwen/examples/`，提供训练与评测 JSONL 模板，解决“仓库结构有了，但数据格式还不统一”的问题。
3. **1.5B / 7B 两套配置**：分别提供 SFT 和 Eval 配置，让 1.5B 用于流程验证，7B 用于主实验。

## 这次训练任务的目标（概括）

这次训练任务的目标不是直接追求最强效果，而是把仓库真正变成一个**能稳定承接 Qwen 微调任务的最小工程闭环**。

换句话说，核心目标是打通：

**数据模板 -> SFT 训练 -> checkpoint 输出 -> 评测生成 -> 结果保存**

这样做的意义是：

- **1.5B** 负责先把训练和评测链路跑通，快速暴露环境、数据和脚本问题。
- **7B** 负责后续正式实验，是主训练模型。
- 评测入口保证你不是“只训练不验证”，而是从一开始就有统一的输出落点。

## 我对这次补丁的定位

这不是最终版训练系统，而是一个**最小但完整的 Qwen 工作流**：

- 现在已经有：训练入口、评测入口、模板数据、模型配置
- 下一阶段最自然的演进方向就是：
  - 补 `DPO` / 偏好数据入口
  - 补更完整的评测指标（不是只看 exact match）
  - 补 80B 教师蒸馏的数据生成链路

## 这次新增的关键文件

- `src/qwen/config.py`
- `src/qwen/data.py`
- `src/qwen/eval.py`
- `src/qwen/cli.py`
- `src/qwen/README.md`
- `configs/qwen_sft_1_5b.json`
- `configs/qwen_eval_1_5b.json`
- `configs/qwen_sft_7b.json`
- `configs/qwen_eval_7b.json`
- `data/qwen/examples/train_messages.example.jsonl`
- `data/qwen/examples/train_instruction.example.jsonl`
- `data/qwen/examples/eval.example.jsonl`

## 运行方式（概括）

训练：

```bash
make qwen-sft-train
```

评测：

```bash
make qwen-sft-eval
```

如果要切到 7B，只需要把 make 变量切到 7B 配置文件即可。

## 当前阶段最重要的结果

你现在的仓库会从“有一些大模型学习代码”提升为：

**已经具备 Qwen 微调 + 基础评测能力的训练仓库雏形。**

这一步的重点不是做大，而是先把结构做对。
