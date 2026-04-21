# llm_learn 仓库 30 天 Hugging Face 训练路线 + 项目包装方案

## 一、当前仓库判断

你当前的 `llm_learn` 已经不是空白学习仓，而是一个正在成型的 Hugging Face 训练主仓：

- 顶层依赖已经覆盖 `transformers`、`datasets`、`peft`、`trl`、`accelerate`、`bitsandbytes`
- 已经有 `qwen-sft` CLI 入口
- 已经形成 **SFT / Eval / DPO** 三段式最小闭环
- `makefile` 也已经把训练与评测做了统一入口
- 数据侧支持 messages / instruction-output / prompt-response / preference 等多种格式

这说明你的下一阶段不该再学“零散工具”，而应该围绕这个仓库做：
1. 基线训练
2. 配置收敛
3. 偏好优化
4. 推理部署
5. 评测分析
6. 简历包装

---

## 二、30 天总目标

30 天后，这个仓库要达到下面的状态：

### 技术目标
- 跑通 `1.5B SFT -> Eval -> DPO -> vLLM 推理验证`
- 至少形成一条稳定可复现实验链路
- 能清楚区分：SFT 提升什么、DPO 提升什么、Eval 如何验证
- 为后续 7B QLoRA 留下配置和实验位

### 项目目标
把仓库包装成一个可以写进简历的综合项目：

**项目名建议：**
`基于 Hugging Face 的 Qwen 指令微调与偏好优化训练框架`

**一句话描述：**
构建了一个支持多格式数据、LoRA/QLoRA SFT、DPO 偏好优化、自动评测与本地推理验证的 Qwen 训练实验框架。

---

## 三、30 天分阶段路线

## 第 1 周：把最小训练闭环跑稳
目标：先不追求效果，先追求链路稳定。

### Day 1
- 用最小样本做 smoke train
- 跑通最小 eval
- 检查输出目录、日志、checkpoint、预测文件

### Day 2
- 整理真实训练集格式
- 把训练数据统一成 1~2 种主格式
- 统计样本数、平均长度、异常样本比例

### Day 3
- 跑 1.5B 正式 SFT baseline
- 记录显存、吞吐、单 step 时长、训练 loss 曲线

### Day 4
- 跑 1.5B eval baseline
- 形成第一版指标表与样例预测

### Day 5
- 抽样看 bad cases
- 分类错误类型：指令不遵循、术语错误、信息缺失、格式问题

### Day 6
- 调一轮 SFT 配置
- 优先调：max_length / learning_rate / grad_acc / train_on_prompt

### Day 7
- 产出第 1 周周报
- 固化 baseline 配置与目录结构

## 第 2 周：把 DPO 偏好优化做出来
目标：从“能训”升级到“能做 post-training”。

### Day 8
- 设计 preference 数据格式规范
- 整理 chosen / rejected 构造方式

### Day 9
- 先做一批小规模 preference 数据
- 确认 prompt/chosen/rejected 能被正常加载

### Day 10
- 用 1.5B SFT adapter 跑第一轮 DPO

### Day 11
- 跑 DPO 后 eval，对比 SFT 前后

### Day 12
- 做第一版对比表：
  - base / sft / dpo
  - 样例回答对比

### Day 13
- 分析 DPO 改善项与退化项
- 判断 preference 数据是否太弱、太脏或太少

### Day 14
- 产出第 2 周周报
- 固化 DPO 数据与配置版本

## 第 3 周：接入 vLLM 与应用验证
目标：把训练成果接到服务层，形成真实可展示项目。

### Day 15
- 用 vLLM 部署 base / sft / dpo 三个版本中的一个

### Day 16
- 写本地推理脚本
- 固定 prompt 模板与测试集

### Day 17
- 对比本地离线 eval 与 vLLM 服务输出是否一致

### Day 18
- 设计一个任务化 demo：
  - 问答
  - 术语解释
  - 指令跟随
  - 简单 reasoning

### Day 19
- 记录服务侧指标：
  - 首 token 延迟
  - 吞吐
  - 显存占用
  - 不同 batch 行为

### Day 20
- 做应用侧 bad cases 分析

### Day 21
- 产出第 3 周周报
- 形成“训练 + 服务”联合展示版本

## 第 4 周：做项目包装与简历材料
目标：把实验变成能讲、能写、能面试的项目。

### Day 22
- 整理仓库目录
- 固定 `configs/ data/ docs/ patch/ reports/` 结构

### Day 23
- 写总 README
- 讲清楚目标、结构、命令、数据格式、实验路线

### Day 24
- 写实验报告：
  - 数据
  - 配置
  - baseline
  - DPO 对比
  - bad case

### Day 25
- 整理图表：
  - loss 曲线
  - 对比表
  - 样例输出表

### Day 26
- 形成项目简历版本 1

### Day 27
- 形成面试讲稿：
  - 为什么这样设计
  - SFT 和 DPO 区别
  - 为什么先做 1.5B
  - 为什么要保留 vLLM 部署层

### Day 28
- 补一轮代码清理和注释

### Day 29
- 形成项目答辩材料 / 演示脚本

### Day 30
- 输出最终版项目包：
  - README
  - 配置
  - 数据样例
  - eval 结果
  - 简历条目
  - 面试话术

---

## 四、你这个项目最终应该包装成什么

## 项目定位
这不是“我学过几个工具”的仓库，而应该包装成：

**一个面向 Qwen 的 Hugging Face 训练实验框架**

## 你要突出的方法能力
- 会做多格式指令数据整理
- 会做 LoRA / QLoRA 监督微调
- 会做 DPO 偏好优化
- 会做自动评测与错误分析
- 会把训练结果挂到 vLLM 服务层验证

## 你要突出的工程能力
- 配置化训练
- 命令行入口
- 数据格式适配
- checkpoint 管理
- 结果目录与实验可复现

---

## 五、建议的简历写法

## 项目名
**基于 Hugging Face 的 Qwen 指令微调与偏好优化训练框架**

## 简历描述（推荐版）
- 基于 Transformers / Datasets / PEFT / TRL 搭建 Qwen 训练实验框架，支持多种指令数据与偏好数据格式的统一加载与配置化训练。
- 实现 Qwen 1.5B 模型的 LoRA/QLoRA SFT、自动评测与 DPO 偏好优化闭环，完成从基线训练到对齐优化的实验流程。
- 接入 vLLM 对训练后模型进行本地推理验证，结合离线评测结果分析模型在指令跟随、术语解释与回答稳定性上的变化。
- 通过配置版本、数据样例、评测输出和 bad case 分析提高实验可复现性，为后续扩展到 7B QLoRA 与更复杂对齐方法预留接口。

---

## 六、第 1 天最合理的补丁策略

这次不给你大改训练逻辑，而是补第 1 天最需要的内容：

1. 最小 SFT smoke 配置
2. 最小 Eval smoke 配置
3. 最小 train / val / eval 示例数据
4. `make` 快捷入口
5. Day 1 执行 runbook

这样做的好处是：
- 当天就能执行
- 不依赖你先准备完整大数据
- 先验证训练主链路
- 后续可以直接作为 smoke regression

---

## 七、第 1 天执行建议

建议你今天只做这 4 步：

```bash
make qwen-day1-train
make qwen-day1-eval
```

然后记录：

- 是否顺利进入训练
- 是否成功保存 checkpoint
- eval 是否能读取 adapter
- 预测输出是否合理
- 显存峰值和 step 时间

只要这 4 件事打通，Day 1 就是成功的。

---

## 八、这轮生成文件

- 长报告：`llm_learn_30day_route_round4_2026-04-21.md`
- 补丁：`llm_learn_day1_patch_2026-04-21.patch`

## 九、校验结果

### 补丁校验
- `git apply --check`: PASS

### JSON / JSONL 校验
{
  "configs/qwen_sft_day1_smoke.json": {
    "ok": true,
    "keys": [
      "model",
      "data",
      "train"
    ]
  },
  "configs/qwen_eval_day1_smoke.json": {
    "ok": true,
    "keys": [
      "model_name_or_path",
      "adapter_path",
      "eval_file",
      "output_dir",
      "max_length",
      "max_new_tokens",
      "temperature",
      "top_p",
      "do_sample",
      "batch_size",
      "load_in_4bit",
      "trust_remote_code",
      "system_prompt",
      "save_predictions_name"
    ]
  },
  "data/qwen/examples/day1_train.jsonl": {
    "ok": true,
    "rows": 6
  },
  "data/qwen/examples/day1_val.jsonl": {
    "ok": true,
    "rows": 2
  },
  "data/qwen/examples/day1_eval.jsonl": {
    "ok": true,
    "rows": 3
  }
}

