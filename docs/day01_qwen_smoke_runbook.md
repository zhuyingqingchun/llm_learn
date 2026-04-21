# Day 1：Qwen SFT smoke run

## 目标
第一天不追求指标，而是追求把 **数据 -> 配置 -> 训练 -> checkpoint -> 评测** 这一条最小链路跑通。

## 新增内容
- `configs/qwen_sft_day1_smoke.json`
- `configs/qwen_eval_day1_smoke.json`
- `data/qwen/examples/day1_train.jsonl`
- `data/qwen/examples/day1_val.jsonl`
- `data/qwen/examples/day1_eval.jsonl`
- `make qwen-day1-train`
- `make qwen-day1-eval`

## 推荐执行顺序
```bash
make qwen-day1-train
make qwen-day1-eval
```

## 成功标准
1. SFT 能正常启动并输出日志。
2. `./qwen_checkpoints/day1_qwen2_5_1_5b_sft_smoke` 下能看到保存结果。
3. Eval 能正常读取 adapter 并生成预测文件。
4. 你能确认一轮最小链路没有路径错误、数据错误、格式错误。

## 第一天重点记录
- 显存占用
- 单 step 耗时
- 是否出现 tokenizer / chat template / pad token 报错
- checkpoint 是否正常保存
- 评测输出是否可读
