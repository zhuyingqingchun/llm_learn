# LLM Learn

大语言模型学习与实验项目。

## 环境配置

### 系统信息

| 项目 | 信息 |
|------|------|
| Python 版本 | 3.13.11 |
| PyTorch 版本 | 2.10.0+cu128 |
| CUDA 版本 | 12.8 |
| GPU 型号 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 单卡显存 | 97887 MiB（约 95.6 GB） |
| GPU 数量 | 4 |
| 总显存 | 约 382 GB（4 × 95.6 GB） |
| 单卡功耗 | 600W |
| 驱动版本 | 580.126.09 |

### 依赖安装

```bash
# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 默认路径

| 用途 | 路径 |
|------|------|
| 数据集路径 | `/mnt/PRO6000_disk/data` |
| 模型路径 | `/mnt/PRO6000_disk/models` |

> 以上路径为默认下载、加载路径。

## 项目结构

```
llm_learn/
├── README.md          # 项目说明文档
└── ...
```

## 已下载数据集

### 文本数据集

| 数据集 | 来源 | 大小 | 状态 |
|--------|------|------|------|
| `AI-ModelScope/alpaca-gpt4-data-zh` | 魔塔社区 | - | 已完成 |
| `hkust-nlp/deita-10k-v0` | 魔塔社区 | - | 已完成 |
| `AI-ModelScope/gsm8k` | 魔塔社区 | - | 已完成 |

### 多模态数据集

| 数据集 | 来源 | 已下载文件 | 大小 | 状态 |
|--------|------|------------|------|------|
| `AI-ModelScope/LLaVA-CoT-100k` | 魔塔社区 (GIT LFS) | `train.jsonl` | 291 MB | 已完成 |
| `modelscope/LLaVA-R1-100k` | 魔塔社区 (GIT LFS) | `train.jsonl` | 766 MB | 已完成 |

> 多模态数据集的图像数据（`image.zip.part-*`、`images.zip`）暂未下载。

### 数据集结构

**LLaVA-CoT-100k**
- 字段：`id`, `image`, `conversations`
- 对话格式：`[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`
- 推理标签：`<SUMMARY>` → `<CAPTION>` → `<REASONING>` → `<CONCLUSION>`

**LLaVA-R1-100k**
- 字段：`id`, `image`, `caption`, `messages`, `solution`, `correctness`, `usage`, `judge_method`, `source`
- 对话格式：`[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- 推理标签：`<think>` → `<answer>`

### 下载注意事项

多模态数据集下载时，若使用代理可能出现 `SSLEOFError` 或 `ChunkedEncodingError`。稳定下载方案：

```bash
# 1. 临时取消代理
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY

# 2. 拉仓库骨架（不下载大文件）
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/datasets/AI-ModelScope/LLaVA-CoT-100k.git

# 3. 查看 LFS 文件
cd LLaVA-CoT-100k && git lfs ls-files

# 4. 按需拉取指定文件
git lfs pull --include="train.jsonl"
```

## 使用说明

待补充。
