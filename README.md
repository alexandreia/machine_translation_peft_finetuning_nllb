# NLLB Experiment

This experiment uses `facebook/nllb-200-distilled-600M`, a multilingual translation model from Hugging Face. The model is evaluated on the same preprocessed WMT18 Czech-English dataset used by the baseline and other experiments. NLLB uses its own tokenizer and language codes as part of the pretrained model pipeline. Since the dataset split and evaluation metrics are shared across experiments, differences in performance mainly reflect the model pipelines rather than preprocessing differences.


## Dataset

All group experiments use the same Hugging Face dataset:

```python
from datasets import load_dataset

ds = load_dataset("charlie0831/wmt18-cs-en-preprocessed")
```

Shared preprocessing from Chien:

- WMT18 Czech-English data
- shuffled with seed `42`
- removed sentence pairs where either Czech or English has fewer than 2 words
- train/dev/test sizes: `48,000 / 1,000 / 1,000`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run NLLB Evaluation

Quick smoke run on 10 examples:

```bash
python experiments/nllb_evaluate.py --split test --max-samples 10
```

Full test evaluation:

```bash
python experiments/nllb_evaluate.py --split test
```

## LoRA Fine-Tuning

LoRA is the parameter-efficient fine-tuning setup for this experiment. It keeps the pretrained NLLB model mostly frozen and trains small added matrices inside attention layers.

Beginner-friendly LoRA script:

```bash
python experiments/train_lora.py
```

This script is organized like a simple sklearn/PyTorch workflow:

```text
1. load dataset
2. load tokenizer and model
3. add LoRA
4. tokenize data
5. train
6. save the LoRA adapter
7. run a small Hugging Face pipeline translation check
```

Quick smoke run with very small data:

```bash
python experiments/train_lora.py \
  --max-train-samples 50 \
  --max-dev-samples 20 \
  --epochs 1 \
  --eval-steps 10 \
  --save-steps 10
```

Main LoRA run:

```bash
python experiments/train_lora.py \
  --max-train-samples 48000 \
  --max-dev-samples 1000 \
  --epochs 1 \
  --learning-rate 5e-4 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.1
```

The trained adapter is saved to:

```text
outputs/nllb-cs-en-lora/adapter
```

Note: Hugging Face pipelines are used here only for quick translation checks after training. Fine-tuning itself uses `Seq2SeqTrainer`, because pipelines are inference helpers and do not train models.

Tiny training smoke run:

```bash
python experiments/nllb_lora_finetune.py \
  --max-train-samples 100 \
  --max-dev-samples 50 \
  --num-train-epochs 1 \
  --eval-steps 25 \
  --save-steps 25
```

Main Czech-English LoRA fine-tuning run:

```bash
python experiments/nllb_lora_finetune.py \
  --num-train-epochs 15 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-steps 100 \
  --eval-steps 100 \
  --save-steps 100
```

Optional LoRA target-module ablation:

```bash
python experiments/nllb_lora_finetune.py \
  --output-dir outputs/nllb-lora-all-attn \
  --lora-target-modules q_proj k_proj v_proj o_proj
```

Evaluate the fine-tuned adapter on the WMT18 test split:

```bash
python experiments/nllb_evaluate.py \
  --split test \
  --adapter outputs/nllb-lora/adapter
```

This gives the core comparison:

```text
NLLB zero-shot
NLLB + LoRA fine-tuning on Czech-English
```

The script saves:

- generated translations as JSONL
- metric summary as JSON

