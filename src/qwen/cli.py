from __future__ import annotations

from pathlib import Path
import json

import torch
import typer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from qwen.config import QwenSFTConfig, load_qwen_sft_config, save_resolved_config
from qwen.data import load_qwen_sft_datasets


app = typer.Typer(help="Qwen 最小 SFT 训练入口")


def _build_quantization_config(config: QwenSFTConfig):
    if not config.model.load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_model_and_tokenizer(config: QwenSFTConfig):
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
        torch_dtype=torch.bfloat16 if config.train.bf16 else None,
    )

    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def _build_training_args(config: QwenSFTConfig, has_eval: bool) -> TrainingArguments:
    evaluation_strategy = "steps" if has_eval else "no"
    report_to = config.train.report_to if config.train.report_to else []

    return TrainingArguments(
        output_dir=config.train.output_dir,
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
        remove_unused_columns=False,
        gradient_checkpointing=config.train.gradient_checkpointing,
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        logging_strategy="steps",
        report_to=report_to,
        ddp_find_unused_parameters=False,
        seed=config.train.seed,
    )


@app.command()
def train(
    config_path: str = typer.Option(
        "./configs/qwen_sft_minimal.json",
        help="Qwen SFT 配置文件路径",
    )
):
    config = load_qwen_sft_config(config_path)
    set_seed(config.train.seed)

    model, tokenizer = _build_model_and_tokenizer(config)
    datasets = load_qwen_sft_datasets(config.data, tokenizer)

    training_args = _build_training_args(
        config,
        has_eval="validation" in datasets and len(datasets["validation"]) > 0,
    )
    save_resolved_config(config, training_args.output_dir)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    summary = {
        "train_size": len(datasets["train"]),
        "validation_size": len(datasets.get("validation", [])),
        "output_dir": training_args.output_dir,
        "model_name_or_path": config.model.model_name_or_path,
    }
    summary_path = Path(training_args.output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
