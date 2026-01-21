"""
Process CF-ARMOR datasets for training
Converts JSONL format to Hugging Face dataset format
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import List, Dict

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def format_for_training(examples: List[Dict], model_name: str = "qwen") -> List[Dict]:
    """
    Format examples for chat model training
    Qwen uses specific chat template format
    """
    formatted = []

    for ex in examples:
        # Handle both 'response' and 'json_response' fields (for malformed examples)
        response = ex.get("response") or ex.get("json_response")

        if not response:
            print(f"Warning: Skipping example without response field")
            continue

        # Handle case where response is a dict (malformed API output)
        if isinstance(response, dict):
            if "response" in response:
                response = response["response"]
            else:
                print(f"Warning: Skipping example with malformed dict response")
                continue

        # Ensure response is a string
        if not isinstance(response, str):
            print(f"Warning: Skipping example with non-string response (type: {type(response)})")
            continue

        # Create chat format
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": response}
        ]

        formatted.append({
            "messages": messages
        })

    return formatted

def process_dataset(
    input_file: str,
    output_dir: str,
    train_split: float = 0.9,
    dataset_name: str = "dataset"
):
    """
    Load, format, and split dataset

    Args:
        input_file: Path to raw JSONL file
        output_dir: Directory to save processed dataset
        train_split: Train/eval split ratio
        dataset_name: Name for the dataset
    """
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")

    # Load raw data
    raw_data = load_jsonl(input_file)
    print(f"Loaded {len(raw_data)} examples from {input_file}")

    # Format for training
    formatted_data = format_for_training(raw_data)
    print(f"Formatted {len(formatted_data)} examples")

    # Create Hugging Face dataset
    dataset = Dataset.from_list(formatted_data)

    # Split train/eval
    split_dataset = dataset.train_test_split(
        test_size=1-train_split,
        seed=42
    )

    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'eval': split_dataset['test']
    })

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"✓ Train: {len(dataset_dict['train'])} examples")
    print(f"✓ Eval: {len(dataset_dict['eval'])} examples")
    print(f"✓ Saved to: {output_path}")

    return dataset_dict

def main():
    """Process all datasets"""
    datasets_to_process = {
        "hybrid": {
            "input": "data/raw/dataset_c_hybrid.jsonl",
            "output": "data/processed/hybrid",
            "description": "CF-ARMOR (Hybrid Reasoning)"
        },
        "intent": {
            "input": "data/raw/dataset_a_intent.jsonl",
            "output": "data/processed/intent",
            "description": "Intent Extraction (ARMOR-style)"
        },
        "consequence": {
            "input": "data/raw/dataset_b_conseq.jsonl",
            "output": "data/processed/consequence",
            "description": "Consequence Simulation"
        },
        "direct": {
            "input": "data/raw/dataset_direct.jsonl",
            "output": "data/processed/direct",
            "description": "Direct Rejection (Baseline)"
        }
    }

    print("\n" + "="*70)
    print("CF-ARMOR Dataset Processing")
    print("="*70)

    processed_datasets = {}

    for name, config in datasets_to_process.items():
        try:
            dataset = process_dataset(
                input_file=config["input"],
                output_dir=config["output"],
                dataset_name=f"{name} - {config['description']}"
            )
            processed_datasets[name] = dataset
        except Exception as e:
            print(f"\n⚠️  Error processing {name}: {e}")
            continue

    print("\n" + "="*70)
    print("✓ Dataset Processing Complete")
    print("="*70)
    print(f"\nProcessed {len(processed_datasets)} datasets:")
    for name in processed_datasets:
        print(f"  • {name}")
    print("\nReady for training!")
    print("  Run: python training/train.py --dataset hybrid")

    return processed_datasets

if __name__ == "__main__":
    main()
