#!/bin/bash
# CF-ARMOR Training Quick Start
# Run this script to train the hybrid CF-ARMOR model

set -e  # Exit on error

echo "========================================================================"
echo "CF-ARMOR Training Quick Start"
echo "========================================================================"
echo ""

# Check if in cf-armor directory
if [ ! -f "training/train.py" ]; then
    echo "Error: Please run this script from the cf-armor directory"
    echo "  cd /Users/yh/Downloads/FYP/20Jan/cf-armor"
    exit 1
fi

# Step 1: Check/Install dependencies
echo "Step 1: Checking dependencies..."
echo "------------------------------------------------------------------------"

if python -c "import torch; import transformers; import peft; import trl" 2>/dev/null; then
    echo "✓ Dependencies already installed"
else
    echo "Installing dependencies (this may take 5-10 minutes)..."
    pip install -q torch transformers datasets peft accelerate bitsandbytes trl sentencepiece protobuf
    echo "✓ Dependencies installed"
fi
echo ""

# Step 2: Process data
echo "Step 2: Processing datasets..."
echo "------------------------------------------------------------------------"

if [ -d "data/processed/hybrid" ]; then
    echo "✓ Data already processed (data/processed/hybrid exists)"
else
    python training/process_data.py
fi
echo ""

# Step 3: Train model
echo "Step 3: Training CF-ARMOR model..."
echo "------------------------------------------------------------------------"
echo "This will take 1-2 hours on GPU"
echo "Model: Qwen/Qwen2.5-3B-Instruct"
echo "Dataset: Hybrid (CF-ARMOR)"
echo "Epochs: 3"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

python training/train.py \
    --dataset hybrid \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4

echo ""
echo "========================================================================"
echo "✓ Training Complete!"
echo "========================================================================"
echo ""
echo "Model saved to: outputs/models/hybrid_*/final/"
echo ""
echo "To test your model:"
echo "  python training/test_model.py --model outputs/models/hybrid_*/final --interactive"
echo ""
