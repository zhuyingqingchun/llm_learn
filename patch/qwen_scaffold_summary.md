# Qwen 最小训练骨架补丁说明

这份补丁做了四件事：

1. 新增 `src/qwen/` 目录，提供最小可运行的 Qwen SFT 训练骨架。
2. 新增 `configs/qwen_sft_minimal.json`，作为训练配置文件入口。
3. 修改 `pyproject.toml`，把 `qwen-sft` 注册为脚本入口，并补上 `peft / accelerate / bitsandbytes` 依赖。
4. 修改 `makefile`，新增 `make qwen-sft-train` 目标。

训练目标不是直接追求最终最强效果，而是先把下面这条链打通：

`指令数据 -> chat template 格式化 -> loss 只监督 assistant 回复 -> LoRA/QLoRA 训练 -> checkpoint 导出`

这条链打通后，仓库就从“Transformer / GPT 学习代码集合”升级成了“可继续扩展到 Qwen SFT / DPO / 蒸馏”的训练仓库。
