# Qwen 最小 SFT / Eval 骨架

这个目录提供了一个最小可运行的 Qwen SFT 训练和评测入口，目标是先把 **数据格式 -> LoRA/QLoRA 微调 -> checkpoint 输出 -> 模型评测** 这条链打通。

## 当前支持的数据格式

### 1. messages 格式
```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"介绍一下你自己"},
  {"role":"assistant","content":"我是一个用于训练实验的 Qwen 模型。"}
]}
```

### 2. instruction / input / output 格式
```json
{"instruction":"解释 Transformer 是什么","input":"","output":"Transformer 是一种基于注意力机制的序列建模架构。"}
```

### 3. prompt / response 格式
```json
{"prompt":"给我一句鼓励的话","response":"继续推进，你已经离可复用训练链路很近了。"}
```

## 最小运行方式

### 训练

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

## 配置文件说明

| 配置文件 | 用途 |
|---------|------|
| `configs/qwen_sft_1_5b.json` | 1.5B 模型 SFT 训练配置 |
| `configs/qwen_eval_1_5b.json` | 1.5B 模型评测配置 |
| `configs/qwen_sft_7b.json` | 7B 模型 SFT 训练配置（4bit 量化） |
| `configs/qwen_eval_7b.json` | 7B 模型评测配置（4bit 量化） |

## 示例数据

`data/qwen/examples/` 目录下提供了训练和评测的示例数据模板。
