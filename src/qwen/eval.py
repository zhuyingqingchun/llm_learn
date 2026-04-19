from __future__ import annotations

from pathlib import Path
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import QwenEvalConfig, save_resolved_config
from .data import load_qwen_eval_dataset


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _build_quantization_config(config: QwenEvalConfig):
    if not config.load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_generation_model_and_tokenizer(config: QwenEvalConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = _build_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)

    model.eval()
    return model, tokenizer


def run_qwen_evaluation(config: QwenEvalConfig) -> dict:
    model, tokenizer = load_generation_model_and_tokenizer(config)
    dataset = load_qwen_eval_dataset(config.eval_file, config.system_prompt)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir, "resolved_qwen_eval_config.json")

    predictions_path = output_dir / config.save_predictions_name
    total = 0
    exact_match_hits = 0

    with predictions_path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            prompt_text = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
            )
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_length = model_inputs["input_ids"].shape[-1]
            output_ids = generated[0][prompt_length:]
            prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            reference = sample.get("reference")

            exact_match = None
            if reference is not None:
                total += 1
                exact_match = int(_normalize_text(prediction) == _normalize_text(reference))
                exact_match_hits += exact_match

            record = {
                "id": sample["id"],
                "prediction": prediction,
                "reference": reference,
                "exact_match": exact_match,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "eval_size": len(dataset),
        "labeled_eval_size": total,
        "exact_match": exact_match_hits / total if total > 0 else None,
        "predictions_path": str(predictions_path),
        "model_name_or_path": config.model_name_or_path,
        "adapter_path": config.adapter_path,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
