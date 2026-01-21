#!/usr/bin/env python3
"""
MLX-based training script for CF-ARMOR on Apple Silicon
Optimized for M1/M2/M3 MacBooks
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def convert_to_mlx_format(dataset_name: str = "hybrid"):
    """
    Convert Hugging Face dataset format to MLX format (JSONL with 'text' field)

    MLX expects JSONL files where each line has a 'text' field with the formatted conversation.
    """
    print("="*70)
    print(f"Converting {dataset_name} dataset to MLX format")
    print("="*70)

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    # Load processed dataset
    dataset_path = f"data/processed/{dataset_name}"
    dataset = load_from_disk(dataset_path)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True
    )

    # Create MLX data directory
    mlx_data_dir = Path(f"data/mlx/{dataset_name}")
    mlx_data_dir.mkdir(parents=True, exist_ok=True)

    # Convert train split
    print(f"\nConverting train split ({len(dataset['train'])} examples)...")
    train_file = mlx_data_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for example in dataset['train']:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            f.write(json.dumps({"text": text}) + "\n")

    # Convert eval split
    print(f"Converting eval split ({len(dataset['eval'])} examples)...")
    eval_file = mlx_data_dir / "valid.jsonl"
    with open(eval_file, 'w') as f:
        for example in dataset['eval']:
            text = tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            f.write(json.dumps({"text": text}) + "\n")

    print(f"✓ MLX format data saved to: {mlx_data_dir}")
    print(f"  - train.jsonl: {len(dataset['train'])} examples")
    print(f"  - valid.jsonl: {len(dataset['eval'])} examples")

    return str(mlx_data_dir)


def train_mlx(
    dataset_name: str = "hybrid",
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
    output_dir: str = "outputs/models_mlx",
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Train CF-ARMOR model using MLX (optimized for Apple Silicon)

    Args:
        dataset_name: Which dataset to use (hybrid, intent, consequence, direct)
        model_name: MLX-compatible model name
        output_dir: Where to save the model
        iters: Number of training iterations
        batch_size: Batch size for training
        learning_rate: Learning rate
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """

    print("\n" + "="*70)
    print(f"CF-ARMOR MLX Training: {dataset_name.upper()}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output: {output_dir}")
    print(f"  Iterations: {iters}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  LoRA dropout: {lora_dropout}")
    print("="*70)

    # Step 1: Convert dataset to MLX format
    print("\n[Step 1/3] Converting dataset to MLX format...")
    try:
        mlx_data_path = convert_to_mlx_format(dataset_name)
    except Exception as e:
        print(f"✗ Error converting dataset: {e}")
        print("\nMake sure you've run: python training/process_data.py")
        return False

    # Step 2: Setup output directory
    print("\n[Step 2/3] Setting up output directory...")
    from datetime import datetime
    experiment_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    adapter_path = Path(output_dir) / experiment_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {adapter_path}")

    # Step 3: Train with MLX
    print("\n[Step 3/3] Training with MLX...")
    print("-" * 70)
    print(f"Using MLX default LoRA parameters (rank=8, alpha=16)")
    print("For custom LoRA params, create a config.yaml file")

    cmd = [
        "python", "-m", "mlx_lm", "lora",  # Fixed: space between mlx_lm and lora
        "--model", model_name,
        "--train",
        "--data", mlx_data_path,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--adapter-path", str(adapter_path),
        "--save-every", "100",
        "--test-batches", "10",
        "--seed", "42",
        "--grad-checkpoint",  # Reduce memory usage
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        import time
        start_time = time.time()

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )

        elapsed_time = time.time() - start_time

        print("\n" + "="*70)
        print("✓ Training Complete!")
        print("="*70)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Adapter saved to: {adapter_path}")
        print(f"\nTo test the model:")
        print(f"  python training/test_mlx.py --model {model_name} --adapter {adapter_path}")
        print("="*70)

        # Save training info
        info = {
            "dataset": dataset_name,
            "model": model_name,
            "iters": iters,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "adapter_path": str(adapter_path),
            "trained_at": datetime.now().isoformat(),
            "framework": "MLX",
            "elapsed_time_minutes": elapsed_time/60
        }

        with open(adapter_path / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train CF-ARMOR with MLX (Apple Silicon optimized)"
    )

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
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="MLX model name (default: mlx-community/Qwen2.5-3B-Instruct-4bit)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=600,
        help="Number of training iterations (default: 600)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CF-ARMOR MLX Training (Apple Silicon Optimized)")
    print("="*70)
    print(f"\nUsing MLX framework for fast M1/M2/M3 training")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    success = train_mlx(
        dataset_name=args.dataset,
        model_name=args.model,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    if not success:
        print("\n✗ Training failed")
        sys.exit(1)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
