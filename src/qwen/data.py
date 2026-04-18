from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import QwenDataConfig


def _normalize_messages(sample: dict, data_config: QwenDataConfig) -> list[dict[str, str]]:
    if "messages" in sample and sample["messages"]:
        messages = sample["messages"]
    elif "conversations" in sample and sample["conversations"]:
        messages = []
        for item in sample["conversations"]:
            role = item.get("role") or item.get("from")
            content = item.get("content") or item.get("value")
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                raise ValueError(f"无法识别的 role: {role}")
            messages.append({"role": role, "content": content})
    elif "instruction" in sample and "output" in sample:
        user_prompt = sample["instruction"]
        if sample.get("input"):
            user_prompt = f"{user_prompt}\n\n{sample['input']}"
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": sample["output"]},
        ]
    elif "prompt" in sample and "response" in sample:
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    else:
        raise ValueError(
            "样本格式不支持。请提供 messages / conversations / instruction-output / prompt-response 之一。"
        )

    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT 样本的最后一条消息必须是 assistant。")

    has_system = any(message["role"] == "system" for message in messages)
    if data_config.system_prompt and not has_system:
        messages = [{"role": "system", "content": data_config.system_prompt}] + messages

    return messages


def _tokenize_sample(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    data_config: QwenDataConfig,
) -> dict:
    messages = _normalize_messages(sample, data_config)
    prompt_messages = messages[:-1]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=data_config.max_length,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=data_config.max_length,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()

    if not data_config.train_on_prompt:
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        for idx in range(prompt_len):
            labels[idx] = -100

    supervised_tokens = sum(label != -100 for label in labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "supervised_tokens": supervised_tokens,
    }


def _load_raw_datasets(data_config: QwenDataConfig) -> DatasetDict:
    train_path = Path(data_config.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")

    data_files = {"train": str(train_path)}
    if data_config.validation_file:
        val_path = Path(data_config.validation_file)
        if not val_path.exists():
            raise FileNotFoundError(f"验证数据不存在: {val_path}")
        data_files["validation"] = str(val_path)

    raw_datasets = load_dataset("json", data_files=data_files)

    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(
            test_size=data_config.validation_split_ratio,
            seed=42,
        )
        raw_datasets = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

    return raw_datasets


def load_qwen_sft_datasets(
    data_config: QwenDataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    raw_datasets = _load_raw_datasets(data_config)
    remove_columns = raw_datasets["train"].column_names

    tokenized = raw_datasets.map(
        lambda sample: _tokenize_sample(sample, tokenizer, data_config),
        remove_columns=remove_columns,
        num_proc=data_config.preprocessing_num_workers,
        desc="Tokenizing Qwen SFT dataset",
    )

    tokenized = tokenized.filter(
        lambda sample: sample["supervised_tokens"] > 0,
        desc="Filtering empty-supervision samples",
    )
    tokenized = tokenized.remove_columns(["supervised_tokens"])
    return tokenized
