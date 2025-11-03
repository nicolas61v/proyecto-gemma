"""
Ejemplo mínimo de script para entrenar adaptadores LoRA usando QLoRA (bitsandbytes + PEFT)

ADVERTENCIA:
- QLoRA requiere GPU y CUDA; en Windows se recomienda WSL2 con GPU passthrough o usar Linux.
- bitsandbytes puede no instalarse o funcionar en Windows nativo.

Uso (ejemplo):
  python qlora_finetune.py --train_file data/train.jsonl --output_dir lora_adapter

El archivo JSONL debe tener objetos con campos: {"prompt": "...", "response": "..."}

Este script es una plantilla y puede necesitar ajustes según el modelo y recursos.
"""

import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

try:
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    bnb = None
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def make_prompt(example):
    return f"User: {example['prompt']}\nAssistant: {example['response']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Archivo jsonl de entrenamiento")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--output_dir", type=str, default="lora_adapter")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    if bnb is None or get_peft_model is None:
        raise RuntimeError("Faltan dependencias (bitsandbytes/peft). Revisa el README_QLoRA.md")

    # Cargar dataset
    ds = load_dataset('json', data_files={'train': args.train_file})
    ds = ds['train'].map(lambda ex: {'text': make_prompt(ex)})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=1024)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=['text'])

    # Cargar modelo en 4-bit para QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        device_map='auto',
        quantization_config=bnb.nn.QuantLinear if hasattr(bnb, 'nn') else None
    )

    model = prepare_model_for_kbit_training(model)

    # Configuración LoRA - valores por defecto razonables
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy='epoch',
        fp16=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
