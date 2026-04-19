from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
from typing import Any


@dataclass
class QwenModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    use_lora: bool = True
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class QwenDataConfig:
    train_file: str = "./data/qwen/train.jsonl"
    validation_file: str | None = "./data/qwen/val.jsonl"
    validation_split_ratio: float = 0.02
    max_length: int = 2048
    preprocessing_num_workers: int = 1
    system_prompt: str = "You are a helpful assistant."
    train_on_prompt: bool = False


@dataclass
class QwenTrainConfig:
    output_dir: str = "./qwen_checkpoints/qwen2_5_1_5b_sft"
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class QwenSFTConfig:
    model: QwenModelConfig = field(default_factory=QwenModelConfig)
    data: QwenDataConfig = field(default_factory=QwenDataConfig)
    train: QwenTrainConfig = field(default_factory=QwenTrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QwenEvalConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    eval_file: str = "./data/qwen/eval.jsonl"
    output_dir: str = "./qwen_eval_outputs/qwen2_5_1_5b"
    max_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    batch_size: int = 1
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    system_prompt: str = "You are a helpful assistant."
    save_predictions_name: str = "predictions.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_dataclass(cls: type, payload: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return cls(**filtered)


def load_qwen_sft_config(config_path: str | Path) -> QwenSFTConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    model_cfg = _build_dataclass(QwenModelConfig, payload.get("model", {}))
    data_cfg = _build_dataclass(QwenDataConfig, payload.get("data", {}))
    train_cfg = _build_dataclass(QwenTrainConfig, payload.get("train", {}))
    return QwenSFTConfig(model=model_cfg, data=data_cfg, train=train_cfg)


def load_qwen_eval_config(config_path: str | Path) -> QwenEvalConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _build_dataclass(QwenEvalConfig, payload)


def save_resolved_config(config: QwenSFTConfig | QwenEvalConfig, output_dir: str | Path, filename: str = "resolved_qwen_sft_config.json") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / filename
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    return config_path
