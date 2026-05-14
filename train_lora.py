#!/usr/bin/env python3
"""Fine-tune NLLB with LoRA for Czech to English translation."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


# constants
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_DATASET = "charlie0831/wmt18-cs-en-preprocessed"

NLLB_CZECH = "ces_Latn"
NLLB_ENGLISH = "eng_Latn"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for NLLB Czech to English translation."
    )

    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="outputs/nllb-cs-en-lora")

    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--max-dev-samples", type=int, default=100)

    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)

    return parser.parse_args()


def get_text_pair(example):
    return {"source": example["translation"]["cs"], "target": example["translation"]["en"]}


def load_parallel_data(dataset_name, max_train_samples=None, max_dev_samples=None):
    dataset = load_dataset(dataset_name)

    train_split = dataset["train"]
    dev_split = dataset["validation"]

    # adding if statements for making a smaller training/dev set when I want a quick test run
    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))

    if max_dev_samples is not None:
        dev_split = dev_split.select(range(min(max_dev_samples, len(dev_split))))

    train_data = Dataset.from_list([get_text_pair(example) for example in train_split])
    dev_data = Dataset.from_list([get_text_pair(example) for example in dev_split])

    return train_data, dev_data


def load_nllb(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=NLLB_CZECH)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def add_lora(model, args):
    '''Notes on cofig:
    lora_r: bigger rank means more LoRA capacity, but also more trainable parameters; 8 is a good start
    lora_alpha: controls how strong the LoRA effect is; i.e 32
    lora_dropout: helps with overfitting
    target_modules: specify where we add LoRA in the model; i.e proiectiile din mecanismul de atentie NLLB ["q_proj", "v_proj"]
    '''
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model 

def preprocess_batch(batch, tokenizer, max_source_length, max_target_length):
    model_inputs = tokenizer(
        batch["source"],
        max_length=max_source_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def tokenize_data(tokenizer, train_data, dev_data, args):
    tokenized_train = train_data.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_data.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
        },
    )

    tokenized_dev = dev_data.map(
        preprocess_batch,
        batched=True,
        remove_columns=dev_data.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
        },
    )

    return tokenized_train, tokenized_dev



def train(model, tokenizer, tokenized_train, tokenized_dev, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    return trainer


def save_adapter(trainer, tokenizer, args):
    adapter_dir = Path(args.output_dir) / "adapter"

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "base_model": args.model_name,
        "dataset": args.dataset_name,
        "source_language": NLLB_CZECH,
        "target_language": NLLB_ENGLISH,
        "max_train_samples": args.max_train_samples,
        "max_dev_samples": args.max_dev_samples,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
    }

    metadata_path = Path(args.output_dir) / "training_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return adapter_dir


def translate_examples(model, tokenizer):
    examples = [
        "Toto je testovací věta.",
        "Studenti dnes pracují na projektu strojového překladu.",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for sentence in examples:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = inputs.to(device)

        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_ENGLISH),
            max_new_tokens=256,
            num_beams=4,
        )

        translation = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]

        print("CS:", sentence)
        print("EN:", translation)
        print()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data = load_parallel_data(
        args.dataset_name,
        args.max_train_samples,
        args.max_dev_samples,
    )

    tokenizer, model = load_nllb(args.model_name)
    model = add_lora(model, args)

    tokenized_train, tokenized_dev = tokenize_data(
        tokenizer,
        train_data,
        dev_data,
        args,
    )

    trainer = train(
        model,
        tokenizer,
        tokenized_train,
        tokenized_dev,
        args,
    )

    adapter_dir = save_adapter(trainer, tokenizer, args)
    print(f"Saved adapter to {adapter_dir}")

    translate_examples(trainer.model, tokenizer)


if __name__ == "__main__":
    main()
