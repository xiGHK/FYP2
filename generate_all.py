#!/usr/bin/env python3
"""
Master script to generate all CF-ARMOR datasets
This is a simplified wrapper around the main generation script
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def main():
    print("=" * 70)
    print("CF-ARMOR Dataset Generation")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("✓ Using OpenRouter API with Gemini model\n")

    # Ask user for number of examples
    print("How many examples per dataset? (recommended: 50-100 for testing, 200-300 for production)")
    num_examples = input("Enter number (default: 50): ").strip()
    num_examples = int(num_examples) if num_examples else 50

    print(f"\nGenerating {num_examples} examples per dataset...")
    print("Note: Direct rejection will have 3x examples for balance\n")

    # Run generation
    script_dir = Path(__file__).parent / "scripts"
    gen_script = script_dir / "generate_data.py"

    cmd = f"python {gen_script} --dataset all --num-examples {num_examples}"

    print(f"Running: {cmd}\n")
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ ALL DATASETS GENERATED")
    print("=" * 70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print(f"  • data/raw/dataset_a_intent.jsonl (~{num_examples} examples)")
    print(f"  • data/raw/dataset_b_conseq.jsonl (~{num_examples} examples)")
    print(f"  • data/raw/dataset_c_hybrid.jsonl (~{num_examples} examples)")
    print(f"  • data/raw/dataset_direct.jsonl (~{num_examples * 3} examples)")
    print("\nNext steps:")
    print("  1. Review samples: head -n 1 data/raw/dataset_*.jsonl | jq .")
    print("  2. Process for training (to be implemented)")
    print("  3. Start training (to be implemented)")

if __name__ == "__main__":
    main()
