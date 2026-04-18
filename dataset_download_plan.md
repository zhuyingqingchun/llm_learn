# Qwen 训练与数据集方案总结（ModelScope / 魔塔社区国内下载版）

## 1. 当前已有模型

你目前已有的模型为：

- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen3-Next-80B-A3B-Instruct-FP8`

当前环境说明：

- Conda 环境：`swtorch12`

---

## 2. 总体策略

现阶段建议采用以下主线：

**1.5B 做流程验证 -> 7B 做主训练与主评测 -> 80B FP8 做教师模型 / 高质量推理服务 / 数据蒸馏，不作为第一阶段主训练对象。**

这样做的原因：

- `1.5B` 适合快速验证训练脚本、数据格式、评测脚本、LoRA/QLoRA 流程。
- `7B` 是当前最适合投入精力的主力模型，适合做 SFT、DPO、领域增强和推理增强。
- `80B FP8` 更适合作为教师模型、蒸馏数据生成器和高质量推理服务，不适合在“数据集刚开始准备”的阶段直接作为主训练对象。

---

## 3. 重要前提：当前这三个模型是文本路线，不是原生视觉多模态训练基座

这一点非常关键。

你现在列出来的：

- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen3-Next-80B-A3B-Instruct-FP8`

都应该视为**文本模型路线**。

### 这意味着什么？

如果你现在直接下载图像、视频、音频等多模态数据：

- **不能直接把图像原始输入喂给这三个模型做原生图文训练**
- 当前这些模型更适合：
  - 文本 SFT
  - 文本 DPO
  - 文本 reasoning 增强
  - 多模态蒸馏后的文本监督训练

### 如果你后面要做真正的图文/视频/音频训练

那就应该换成下面这类基座：

- `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL`
- `Qwen2.5-Omni`

也就是说：

- **当前模型可以先做“多模态转文本蒸馏”**
- **如果要做真正的视觉指令微调，就要切到 VL / Omni 路线**

---

## 4. 下载原则：统一使用魔塔社区（ModelScope）国内下载

后续数据集下载统一要求：

- **只走魔塔社区 / ModelScope 国内下载**
- **不走 Hugging Face**
- **不依赖国外网络**
- **不使用你的 VPN 进行数据集获取**
- **命令、脚本、SDK 全部优先写成 ModelScope 版本**

### 安装方式

```bash
pip install -U modelscope
```

---

## 5. 文本数据集方案

文本数据集按三个阶段组织：

- 第一阶段：SFT 起步与流程验证
- 第二阶段：DPO / 偏好优化
- 第三阶段：Reasoning 增强

---

### 5.1 第一阶段：优先下载的文本数据集

#### （1）`AI-ModelScope/alpaca-gpt4-data-zh`
用途：

- 中文指令精调
- 1.5B 流程验证
- 7B 第一版 SFT 基线

特点：

- 适合快速起步
- 适合先把训练链路跑通

---

#### （2）`hkust-nlp/deita-10k-v0`
用途：

- 高质量小规模 SFT
- 作为“少而精”的主训练起点

特点：

- 不大，但质量较高
- 非常适合第一阶段主实验

---

#### （3）`AI-ModelScope/gsm8k`
用途：

- 作为基础评测集
- 做训练前 / 训练后的推理能力对比

特点：

- 适合做 sanity check
- 不建议作为唯一主训练集

---

### 5.2 第二阶段：SFT 跑通后再下载的文本数据集

#### （4）`OpenBMB/UltraFeedback`
用途：

- DPO
- 偏好优化
- chosen / rejected 对构造
- 奖励建模前处理

特点：

- 非常适合回答风格优化和偏好学习

---

#### （5）`swift/Infinity-Instruct`
用途：

- 大规模指令混合扩容
- 后续筛子集训练

特点：

- 体量大
- 不适合一开始全量直接下载并训练
- 更适合后期筛选和抽样使用

---

### 5.3 第三阶段：只有确定做 reasoning 时再下载的文本数据集

#### （6）`open-r1/OpenR1-Math-220k`
用途：

- 数学推理增强
- reasoning 蒸馏
- 作为后续 RL / GRPO 前的 warmup 数据

特点：

- 适合数学与推理任务
- 不建议在第一阶段就使用

---

#### （7）`AI-ModelScope/Mixture-of-Thoughts`
用途：

- step-by-step reasoning 训练
- reasoning SFT
- 多领域思维链训练

特点：

- 更适合推理增强阶段

---

#### （8）`AI-ModelScope/AM-DeepSeek-R1-Distilled-1.4M`
用途：

- 大规模通用 reasoning 增强
- 后期抽样或筛选使用

特点：

- 数据量很大
- 不适合刚开始就全量使用

---

## 6. 文本数据集明确下载顺序

### 现在立刻下载

```text
AI-ModelScope/alpaca-gpt4-data-zh
hkust-nlp/deita-10k-v0
AI-ModelScope/gsm8k
```

### SFT 跑通后再下载

```text
OpenBMB/UltraFeedback
swift/Infinity-Instruct
```

### 确定做 reasoning 后再下载

```text
open-r1/OpenR1-Math-220k
AI-ModelScope/Mixture-of-Thoughts
AI-ModelScope/AM-DeepSeek-R1-Distilled-1.4M
```

---

## 7. 多模态数据集方案

多模态数据集需要分成两条路线来看。

---

### 路线 A：暂时不换模型，继续使用当前文本 Qwen

这条路线下，多模态数据**不能直接用于原生视觉训练**，但依然有用。

使用方法：

- 图片 -> 文本描述 / OCR 结果 / 问答文本
- 视频 -> 关键帧描述 / 视频摘要 / 视频问答文本
- 音频 -> ASR 转写 / 事件描述 / 音频问答文本

本质上是在做：

**多模态蒸馏到文本模型**

也就是说：

- 用多模态数据生成文本监督
- 再喂给 `Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-7B-Instruct`

---

### 7.1 当前文本路线优先下载的多模态数据集

#### （1）`lmms-lab/DocVQA`
用途：

- 文档理解
- 文档图像 -> 文本问答蒸馏
- OCR / 文档问答文本监督

---

#### （2）`lmms-lab/textvqa`
用途：

- 场景图像中的文本理解
- OCR + VQA 类任务蒸馏

---

#### （3）`swift/OCR-VQA`
用途：

- OCR 问答
- 图像文本抽取后转文本监督

---

#### （4）`lmms-lab/ChartQA`
用途：

- 图表理解
- 图表问答转文本监督

---

#### （5）`AI-ModelScope/LLaVA-CoT-100k`
用途：

- 多模态思维链数据
- 图像问题 -> 文本推理答案蒸馏

---

#### （6）`modelscope/LLaVA-R1-100k`
用途：

- 更偏 reasoning 风格的多模态会话蒸馏
- 图像任务转文本推理监督

---

### 7.2 当前文本路线下的多模态数据集下载顺序

#### 现在就可下载

```text
lmms-lab/DocVQA
lmms-lab/textvqa
swift/OCR-VQA
lmms-lab/ChartQA
AI-ModelScope/LLaVA-CoT-100k
modelscope/LLaVA-R1-100k
```

这些数据更适合当前阶段做：

- 多模态转文本监督
- 文本蒸馏
- 文本 reasoning 增强

---

### 路线 B：如果后续切到真正的 VL / Omni 模型

如果你后面决定切换到：

- `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL`
- `Qwen2.5-Omni`

那就可以正式进入视觉指令微调路线。

---

### 7.3 真正 VL 路线优先下载的数据集

#### （1）`AI-ModelScope/LLaVA-Instruct-150K`
用途：

- 视觉 instruction tuning 基础数据
- 图文问答与图文指令训练

---

#### （2）`AI-ModelScope/LLaVA-Pretrain`
用途：

- 视觉预训练 / 对齐阶段数据

---

#### （3）`AI-ModelScope/ShareGPT4V`
用途：

- 图像/视频 caption
- 视觉语言增强

---

#### （4）`ZhipuAI/CogVLM-SFT-311K`
用途：

- 双语视觉指令微调
- 适合中英文混合多模态训练

---

#### （5）`opencsg/llava-instruct-zh-600k`
用途：

- 中文视觉指令微调
- 如果你后面重点做中文 VL，这个非常值得准备

---

### 7.4 体量更大的多模态混合集（后期再考虑）

#### （6）`AI-ModelScope/M3IT`
#### （7）`AI-ModelScope/PangeaInstruct`
#### （8）`AI-ModelScope/the_cauldron`

这些数据集特点：

- 很大
- 很杂
- 清洗成本高
- 更适合后期稳定后筛子集使用

**不建议现在全量直接下载和训练。**

---

## 8. 多模态数据集的明确策略

### 当前阶段（不换基座）

优先策略：

**把多模态数据当作“蒸馏来源”和“文本监督来源”**

也就是说：

- 不直接训练图像输入
- 而是先转成文本格式
- 再继续训练当前的文本 Qwen 模型

最适合当前阶段的数据集：

```text
lmms-lab/DocVQA
lmms-lab/textvqa
swift/OCR-VQA
lmms-lab/ChartQA
AI-ModelScope/LLaVA-CoT-100k
modelscope/LLaVA-R1-100k
```

---

### 后续阶段（切到 VL / Omni）

如果决定真正做图像、视频、音频输入训练，再切到：

- `Qwen2.5-VL-7B-Instruct`
- 或 `Qwen2.5-Omni`

然后优先准备：

```text
AI-ModelScope/LLaVA-Instruct-150K
AI-ModelScope/LLaVA-Pretrain
AI-ModelScope/ShareGPT4V
ZhipuAI/CogVLM-SFT-311K
opencsg/llava-instruct-zh-600k
```

---

## 9. 训练路线建议

现阶段建议训练顺序固定为：

**SFT -> DPO -> （可选）GRPO / reasoning 强化**

### 阶段 A：SFT
用途：

- 打通训练链路
- 验证数据格式
- 建立 baseline

### 阶段 B：DPO
用途：

- 优化回答风格
- 提升偏好一致性
- 改善拒答边界与回答质量

### 阶段 C：GRPO / reasoning 强化
用途：

- 后期增强 reasoning 能力
- 需要更稳定的数据和更明确的奖励设计
- 不建议现在立即开始

---

## 10. 三个模型的职责分工

### `Qwen2.5-1.5B-Instruct`
定位：

- 流程验证模型
- 小样本快速实验模型
- 蒸馏学生模型

用途：

- 验证训练代码
- 试超参数
- 验证数据格式和 loss 曲线

---

### `Qwen2.5-7B-Instruct`
定位：

- 主训练模型
- 主评测模型
- 第一阶段最值得投入的模型

用途：

- SFT 主训练
- DPO 主训练
- 形成第一版可用模型

---

### `Qwen3-Next-80B-A3B-Instruct-FP8`
定位：

- 教师模型
- 高质量标注器
- 线上高质量推理服务

用途：

- 生成高质量回答
- 构造 chosen / rejected 偏好对
- 给 7B / 1.5B 做蒸馏数据
- 高质量推理服务

不建议：

- 在当前阶段把它作为主训练对象

---

## 11. 下载方式：统一使用 ModelScope SDK 或命令行

### 11.1 SDK 示例

```python
from modelscope.msdatasets import MsDataset

# 默认下载路径
DATA_DIR = "/mnt/PRO6000_disk/data"

datasets_to_prepare = [
    "AI-ModelScope/alpaca-gpt4-data-zh",
    "hkust-nlp/deita-10k-v0",
    "AI-ModelScope/gsm8k",
]

for ds_id in datasets_to_prepare:
    ds = MsDataset.load(ds_id, cache_dir=DATA_DIR)
    print(ds_id, "loaded")
```

---

### 11.2 多模态数据 SDK 示例

```python
from modelscope.msdatasets import MsDataset

# 默认下载路径
DATA_DIR = "/mnt/PRO6000_disk/data"

datasets_to_prepare = [
    "lmms-lab/DocVQA",
    "lmms-lab/textvqa",
    "swift/OCR-VQA",
    "lmms-lab/ChartQA",
    "AI-ModelScope/LLaVA-CoT-100k",
    "modelscope/LLaVA-R1-100k",
]

for ds_id in datasets_to_prepare:
    ds = MsDataset.load(ds_id, cache_dir=DATA_DIR)
    print(ds_id, "loaded")
```

---

### 11.3 命令行示例

```bash
pip install -U modelscope

# 默认下载路径: /mnt/PRO6000_disk/data

modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-zh --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset hkust-nlp/deita-10k-v0 --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /mnt/PRO6000_disk/data

modelscope download --dataset lmms-lab/DocVQA --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset lmms-lab/textvqa --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset swift/OCR-VQA --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset lmms-lab/ChartQA --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset AI-ModelScope/LLaVA-CoT-100k --local_dir /mnt/PRO6000_disk/data
modelscope download --dataset modelscope/LLaVA-R1-100k --local_dir /mnt/PRO6000_disk/data
```

---

## 12. 当前最推荐的实际执行顺序

### 第一步：先下文本起步集
```text
AI-ModelScope/alpaca-gpt4-data-zh
hkust-nlp/deita-10k-v0
AI-ModelScope/gsm8k
```

### 第二步：再下当前可用的多模态蒸馏集
```text
lmms-lab/DocVQA
lmms-lab/textvqa
swift/OCR-VQA
lmms-lab/ChartQA
AI-ModelScope/LLaVA-CoT-100k
modelscope/LLaVA-R1-100k
```

### 第三步：SFT 跑稳后补 DPO
```text
OpenBMB/UltraFeedback
```

### 第四步：确定做 reasoning 再补 reasoning 数据
```text
open-r1/OpenR1-Math-220k
AI-ModelScope/Mixture-of-Thoughts
```

### 第五步：如果后续切到 VL / Omni，再准备真正视觉微调数据
```text
AI-ModelScope/LLaVA-Instruct-150K
AI-ModelScope/LLaVA-Pretrain
AI-ModelScope/ShareGPT4V
ZhipuAI/CogVLM-SFT-311K
opencsg/llava-instruct-zh-600k
```

---

## 13. 最终结论

当前最合理的路线是：

### 模型路线
- `1.5B`：流程验证
- `7B`：主训练
- `80B FP8`：教师模型 / 推理服务

### 数据路线
- 先下小规模高质量文本数据
- 再下适合蒸馏成文本监督的多模态数据
- SFT 跑稳后再做 DPO
- 真正需要图像原生训练时，再切到 VL / Omni 模型

### 下载原则
- **统一使用魔塔社区 / ModelScope 国内下载**
- **不走国外**
- **不依赖你的 VPN**

---

## 14. 下一步建议

在这份方案基础上，下一步建议继续做两件事：

1. 把所有下载好的数据集统一整理成一个本地目录结构  
2. 再写一套统一的数据转换脚本，把：
   - 文本 SFT 数据
   - DPO 偏好数据
   - 多模态蒸馏后的文本数据
   统一转成你后续训练脚本可直接使用的格式

建议下一份文件继续写成：

- `dataset_download_plan.md`（本文件）
- `dataset_format_plan.md`
- `qwen_sft_dpo_train_plan.md`
