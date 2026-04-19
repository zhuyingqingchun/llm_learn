# 默认参数（可通过 make VAR=value 覆盖）
PROCESSED_PATH ?= ./data/processed_blocks
MODEL_NAME ?= gpt2
BATCH_SIZE ?= 4
SEQ_LEN ?= 1024
TOKENIZER_PATH ?= gpt2

QWEN_SFT_CONFIG ?= ./configs/qwen_sft_1_5b.json
QWEN_EVAL_CONFIG ?= ./configs/qwen_eval_1_5b.json

.PHONY: install
install:
	uv pip install -e .
	bash -c "source $(PWD)/.venv/bin/activate && \
	transformer-train --install-completion && \
	transformer-interactive --install-completion && \
	qwen-sft --install-completion"

.PHONY: gpt-offline
gpt-offline:
	uv run python -m src.gpt.cli \
		--model-name $(MODEL_NAME) \
		--processed-path $(PROCESSED_PATH) \
		--tokenizer-path $(TOKENIZER_PATH) \
		--batch-size $(BATCH_SIZE) \
		--seq-len $(SEQ_LEN)

.PHONY: qwen-sft-train
qwen-sft-train:
	uv run qwen-sft train --config-path $(QWEN_SFT_CONFIG)

.PHONY: qwen-sft-eval
qwen-sft-eval:
	uv run qwen-sft evaluate --config-path $(QWEN_EVAL_CONFIG)
