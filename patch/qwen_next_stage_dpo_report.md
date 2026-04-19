# llm_learn 下一阶段补丁概括

## 这次补丁补了什么

这次不再补 SFT / Eval，而是进入 **DPO 偏好优化阶段**，主要新增：

- `src/qwen/dpo.py`
  - 最小 DPO 训练入口
  - 支持从 SFT adapter 继续训练
  - 支持 LoRA / QLoRA
  - 输出 `run_summary.json`

- `src/qwen/config.py`
  - 新增 `QwenDPOConfig`
  - 新增 `QwenDPODataConfig`
  - 新增 `QwenDPOTrainConfig`
  - 新增 `load_qwen_dpo_config()`

- `src/qwen/data.py`
  - 新增 `load_qwen_dpo_datasets()`
  - 支持：
    - `prompt / chosen / rejected`
    - `instruction / input / chosen / rejected`
    - `messages / chosen / rejected`

- `src/qwen/cli.py`
  - 新增 `qwen-sft dpo-train --config-path ...`

- `makefile`
  - 新增 `make qwen-dpo-train`

- `pyproject.toml`
  - 新增 `trl` 依赖

- 新增配置文件
  - `configs/qwen_dpo_1_5b.json`
  - `configs/qwen_dpo_7b.json`

- 新增示例数据
  - `data/qwen/examples/preference.example.jsonl`
  - `data/qwen/examples/preference_messages.example.jsonl`

## 这阶段训练任务的目标

这一阶段的目标不是继续验证 SFT 是否能跑，而是把仓库从：

**SFT / Eval 最小闭环**

推进到：

**SFT -> DPO 偏好对齐闭环**

核心目标有三点：

1. 让 SFT 产出的 adapter 可以继续做偏好优化  
2. 让仓库支持标准化 preference 数据格式  
3. 为后续 7B 主实验和教师蒸馏留出统一入口  

## 推荐使用顺序

1. 先用 `qwen_sft_1_5b.json` 跑通 SFT  
2. 再用 `qwen_eval_1_5b.json` 做基础评测  
3. 确认 adapter 正常后，用 `qwen_dpo_1_5b.json` 跑 DPO  
4. 最后再切到 `qwen_dpo_7b.json`

## 说明

这版补丁尽量保持“最小可落地”：
- 不引入复杂 reward model
- 不直接补 GRPO
- 不直接补 80B 教师蒸馏
- 先把 preference 对齐这一层搭起来
