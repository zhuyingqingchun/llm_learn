from __future__ import annotations

from pathlib import Path
import json

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer

from .config import QwenDPOConfig, save_resolved_config
from .data import load_qwen_dpo_datasets


def _build_quantization_config(config: QwenDPOConfig):
    if not config.model.load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_model_tokenizer_and_peft(config: QwenDPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = _build_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config.train.bf16 and torch.cuda.is_available() else None,
    )

    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if config.model.source_adapter_path:
        model = PeftModel.from_pretrained(model, config.model.source_adapter_path)

    if config.model.use_lora:
        if config.model.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    return model, tokenizer, peft_config


def run_qwen_dpo_training(config: QwenDPOConfig) -> dict:
    set_seed(config.train.seed)

    model, tokenizer, peft_config = _build_model_tokenizer_and_peft(config)
    datasets = load_qwen_dpo_datasets(config.data, tokenizer)
    has_eval = "validation" in datasets and len(datasets["validation"]) > 0

    report_to = config.train.report_to if config.train.report_to else []
    training_args = DPOConfig(
        output_dir=config.train.output_dir,
        beta=config.train.beta,
        num_train_epochs=config.train.num_train_epochs,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        warmup_ratio=config.train.warmup_ratio,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        per_device_eval_batch_size=config.train.per_device_eval_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        eval_steps=config.train.eval_steps,
        save_total_limit=config.train.save_total_limit,
        bf16=config.train.bf16,
        fp16=config.train.fp16,
        gradient_checkpointing=config.train.gradient_checkpointing,
        seed=config.train.seed,
        report_to=report_to,
        save_strategy="steps",
        logging_strategy="steps",
        eval_strategy="steps" if has_eval else "no",
        remove_unused_columns=False,
        max_length=config.data.max_prompt_length + config.data.max_completion_length,
        dataset_num_proc=config.data.preprocessing_num_workers,
    )

    save_resolved_config(config, training_args.output_dir, "resolved_qwen_dpo_config.json")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    summary = {
        "train_size": len(datasets["train"]),
        "validation_size": len(datasets.get("validation", [])),
        "output_dir": training_args.output_dir,
        "model_name_or_path": config.model.model_name_or_path,
        "source_adapter_path": config.model.source_adapter_path,
        "beta": config.train.beta,
    }
    summary_path = Path(training_args.output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
