"""
Simple training script for CF-ARMOR
Uses LoRA fine-tuning on Qwen3-4B-Instruct
"""

import os
import torch
import argparse
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
    HAS_COMPLETION_COLLATOR = True
except ImportError:
    HAS_COMPLETION_COLLATOR = False

def setup_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Load model and tokenizer with quantization
    """
    print(f"\n{'='*70}")
    print("Loading Model & Tokenizer")
    print(f"{'='*70}")
    print(f"Model: {model_name}")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                    # Rank - lower for faster training
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.05,
        target_modules=[          # Qwen3 attention modules
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    print(f"\n{'='*70}")
    print("Model Configuration")
    print(f"{'='*70}")
    model.print_trainable_parameters()

    return model, tokenizer

def formatting_prompts_func(example, tokenizer):
    """Format examples using chat template"""
    output_texts = []
    for messages in example["messages"]:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        output_texts.append(text)
    return output_texts

def train(
    dataset_name: str = "hybrid",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "outputs/models",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """
    Train CF-ARMOR model

    Args:
        dataset_name: Which dataset to use (hybrid, intent, consequence, direct)
        model_name: Hugging Face model name
        output_dir: Where to save the model
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
    """
    print("\n" + "="*70)
    print(f"CF-ARMOR Training: {dataset_name.upper()}")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load processed dataset
    dataset_path = f"data/processed/{dataset_name}"
    print(f"Loading dataset from: {dataset_path}")

    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset:")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Eval: {len(dataset['eval'])} examples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print(f"\nMake sure to run data processing first:")
        print(f"  python training/process_data.py")
        return

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Output directory
    experiment_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full_output_dir = f"{output_dir}/{experiment_name}"
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"\nOutput directory: {full_output_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=full_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Effective batch size = 2*8 = 16
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=f"outputs/logs/{experiment_name}",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        evaluation_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=42,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Data collator for chat completion (optional, for newer trl versions)
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "args": training_args,
        "train_dataset": dataset['train'],
        "eval_dataset": dataset['eval'],
        "formatting_func": lambda x: formatting_prompts_func(x, tokenizer),
        "max_seq_length": max_seq_length,
        "packing": False,
    }

    if HAS_COMPLETION_COLLATOR:
        print("Using DataCollatorForCompletionOnlyLM")
        response_template = "<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer
        )
        trainer_kwargs["data_collator"] = collator
    else:
        print("Using default data collator (trl version < 0.8.0)")

    # Trainer
    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * 8})")
    print(f"Learning rate: {learning_rate}")
    print(f"Max sequence length: {max_seq_length}\n")

    try:
        trainer.train()
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        return

    # Save final model
    print("\n" + "="*70)
    print("Saving Model")
    print("="*70)
    final_output = f"{full_output_dir}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print(f"✓ Model saved to: {final_output}")

    # Save training info
    info = {
        "dataset": dataset_name,
        "model": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_examples": len(dataset['train']),
        "eval_examples": len(dataset['eval']),
        "output_dir": final_output,
        "trained_at": datetime.now().isoformat()
    }

    import json
    with open(f"{full_output_dir}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*70)
    print("✓ Training Complete")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel location: {final_output}")
    print(f"\nTo test the model:")
    print(f"  python training/test_model.py --model {final_output}")

def main():
    parser = argparse.ArgumentParser(description="Train CF-ARMOR model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hybrid",
        choices=['hybrid', 'intent', 'consequence', 'direct'],
        help="Which dataset to train on (default: hybrid)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name from Hugging Face"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )

    args = parser.parse_args()

    # Train
    train(
        dataset_name=args.dataset,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length
    )

if __name__ == "__main__":
    main()
