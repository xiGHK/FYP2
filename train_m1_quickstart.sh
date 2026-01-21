#!/bin/bash
# CF-ARMOR MLX Training Quick Start for M1/M2/M3 MacBooks
# Optimized for Apple Silicon

set -e  # Exit on error

echo "========================================================================"
echo "CF-ARMOR MLX Training (Apple Silicon Optimized)"
echo "========================================================================"
echo ""

# Check if in cf-armor directory
if [ ! -f "training/train_mlx.py" ]; then
    echo "Error: Please run this script from the cf-armor directory"
    echo "  cd /Users/yh/Downloads/FYP/20Jan/cf-armor"
    exit 1
fi

# Step 1: Check/Install MLX dependencies
echo "Step 1: Checking MLX dependencies..."
echo "------------------------------------------------------------------------"

if python -c "import mlx.core; import mlx_lm" 2>/dev/null; then
    echo "✓ MLX already installed"
else
    echo "Installing MLX dependencies (this may take 2-3 minutes)..."
    pip install -q mlx mlx-lm transformers datasets sentencepiece protobuf
    echo "✓ MLX dependencies installed"
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

# Step 3: Train model with MLX
echo "Step 3: Training CF-ARMOR model with MLX..."
echo "------------------------------------------------------------------------"
echo "This will take ~15-20 minutes on M1"
echo "Model: mlx-community/Qwen2.5-3B-Instruct-4bit (optimized for Apple Silicon)"
echo "Dataset: Hybrid (CF-ARMOR)"
echo "Iterations: 600 (~3 epochs)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

python training/train_mlx.py \
    --dataset hybrid \
    --iters 600 \
    --batch-size 4 \
    --lr 1e-5

echo ""
echo "========================================================================"
echo "✓ Training Complete!"
echo "========================================================================"
echo ""
echo "Model saved to: outputs/models_mlx/hybrid_*/adapters.safetensors"
echo ""
echo "To test your model:"
echo "  python training/test_mlx.py \\"
echo "    --model mlx-community/Qwen2.5-3B-Instruct-4bit \\"
echo "    --adapter outputs/models_mlx/hybrid_* \\"
echo "    --interactive"
echo ""
