#!/usr/bin/env python3
"""Evaluate an NLLB model on the shared WMT18 Czech-English dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import evaluate
import torch
from peft import PeftModel
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed


DATASET_NAME = "charlie0831/wmt18-cs-en-preprocessed"
MODEL_NAME = "facebook/nllb-200-distilled-600M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config, e.g. ces_Latn-eng_Latn for facebook/flores.",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "devtest", "validation", "test"])
    parser.add_argument("--source-lang", default="cs", help="Dataset source language key.")
    parser.add_argument("--target-lang", default="en", help="Dataset target language key.")
    parser.add_argument("--nllb-source-lang", default="ces_Latn", help="NLLB source language code.")
    parser.add_argument("--nllb-target-lang", default="eng_Latn", help="NLLB target language code.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/nllb")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_split_name(split: str) -> str:
    return "validation" if split == "dev" else split


def get_pair(example: dict[str, Any], source_lang: str, target_lang: str) -> tuple[str, str]:
    """Support either {'translation': {'cs': ..., 'en': ...}} or flat JSON fields."""
    if "translation" in example and isinstance(example["translation"], dict):
        translation = example["translation"]
        return str(translation[source_lang]), str(translation[target_lang])

    possible_source_keys = [
        source_lang,
        "source",
        "src",
        f"{source_lang}_text",
        f"sentence_{source_lang}",
    ]
    possible_target_keys = [
        target_lang,
        "target",
        "tgt",
        f"{target_lang}_text",
        f"sentence_{target_lang}",
    ]

    source = next((example[key] for key in possible_source_keys if key in example), None)
    target = next((example[key] for key in possible_target_keys if key in example), None)
    if source is None or target is None:
        raise KeyError(
            "Could not find source/target fields. Expected a translation dict, "
            f"or source keys {possible_source_keys} and target keys {possible_target_keys}. "
            f"Available keys: {list(example.keys())}"
        )
    return str(source), str(target)


def batched(items: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = normalize_split_name(args.split)
    dataset_dict = load_dataset(args.dataset, args.dataset_config) if args.dataset_config else load_dataset(args.dataset)
    if args.split == "dev" and split not in dataset_dict and "dev" in dataset_dict:
        split = "dev"
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    pairs = [get_pair(example, args.source_lang, args.target_lang) for example in dataset]
    sources = [source for source, _ in pairs]
    references = [target for _, target in pairs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=args.nllb_source_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.nllb_target_lang)
    predictions: list[str] = []

    with torch.inference_mode():
        for batch in tqdm(batched(pairs, args.batch_size), desc="Translating"):
            batch_sources = [source for source, _ in batch]
            inputs = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    chrf_result = chrf.compute(predictions=predictions, references=references)

    run_name = f"{args.model.split('/')[-1]}_{split}_{len(predictions)}"
    translations_path = output_dir / f"{run_name}.jsonl"
    metrics_path = output_dir / f"{run_name}_metrics.json"

    with translations_path.open("w", encoding="utf-8") as file:
        for source, reference, prediction in zip(sources, references, predictions, strict=True):
            file.write(
                json.dumps(
                    {"source": source, "reference": reference, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "model": args.model,
        "adapter": args.adapter,
        "split": split,
        "num_examples": len(predictions),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "nllb_source_lang": args.nllb_source_lang,
        "nllb_target_lang": args.nllb_target_lang,
        "seed": args.seed,
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved translations to {translations_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
