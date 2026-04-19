# Qwen 最小 SFT / Eval / DPO 骨架

这个目录提供了一个最小可运行的 Qwen **SFT / Eval / DPO** 入口，目标是把 **数据格式 -> LoRA/QLoRA 微调 -> checkpoint 输出 -> 模型评测 -> 偏好对齐** 这条链逐步打通。

## 当前支持的数据格式

### 1. SFT: messages 格式
```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"介绍一下你自己"},
  {"role":"assistant","content":"我是一个用于训练实验的 Qwen 模型。"}
]}
```

### 2. SFT: instruction / input / output 格式
```json
{"instruction":"解释 Transformer 是什么","input":"","output":"Transformer 是一种基于注意力机制的序列建模架构。"}
```

### 3. SFT: prompt / response 格式
```json
{"prompt":"给我一句鼓励的话","response":"继续推进，你已经离可复用训练链路很近了。"}
```

### 4. DPO: preference 格式
```json
{"prompt":"解释什么是 LoRA。","chosen":"LoRA 是一种低秩适配方法，只训练少量新增参数。","rejected":"LoRA 就是直接做全参数训练。"}
```

也支持：
- `instruction / input / chosen / rejected`
- `messages / chosen / rejected`
- `prompt(消息列表) / chosen / rejected`

## 最小运行方式

### SFT 训练
```bash
make qwen-sft-train
```

或者：

```bash
uv run qwen-sft train --config-path ./configs/qwen_sft_1_5b.json
```

### 评测
```bash
make qwen-sft-eval
```

或者：

```bash
uv run qwen-sft evaluate --config-path ./configs/qwen_eval_1_5b.json
```

### DPO 训练
```bash
make qwen-dpo-train
```

或者：

```bash
uv run qwen-sft dpo-train --config-path ./configs/qwen_dpo_1_5b.json
```

## 配置文件说明

| 配置文件 | 用途 |
|---------|------|
| `configs/qwen_sft_1_5b.json` | 1.5B 模型 SFT 训练配置 |
| `configs/qwen_eval_1_5b.json` | 1.5B 模型评测配置 |
| `configs/qwen_dpo_1_5b.json` | 1.5B 模型 DPO 训练配置 |
| `configs/qwen_sft_7b.json` | 7B 模型 SFT 训练配置（4bit 量化） |
| `configs/qwen_eval_7b.json` | 7B 模型评测配置（4bit 量化） |
| `configs/qwen_dpo_7b.json` | 7B 模型 DPO 训练配置（4bit 量化） |

## 推荐流程

1. 先用 `qwen_sft_1_5b.json` 跑通 SFT。
2. 再用 `qwen_eval_1_5b.json` 做基础评测。
3. 确认 SFT adapter 正常后，再把 `source_adapter_path` 写入 `qwen_dpo_1_5b.json`，开始 DPO。
4. 最后再切到 7B 配置。

## 示例数据

`data/qwen/examples/` 目录下提供了训练、评测和偏好样本模板。
