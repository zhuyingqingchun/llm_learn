# Qwen 最小 SFT 骨架

这个目录提供了一个最小可运行的 Qwen SFT 训练入口，目标是先把 **数据格式 -> LoRA/QLoRA 微调 -> checkpoint 输出** 这条链打通。

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

```bash
make qwen-sft-train
```

或者：

```bash
uv run qwen-sft train --config-path ./configs/qwen_sft_minimal.json
```
