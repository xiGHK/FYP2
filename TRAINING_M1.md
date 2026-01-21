# CF-ARMOR Training on M1/M2/M3 MacBook

Quick guide for training CF-ARMOR on Apple Silicon Macs using MLX.

---

## Why MLX Instead of PyTorch?

MLX is Apple's machine learning framework optimized for Apple Silicon:
- **3-5x faster** than PyTorch on M1/M2/M3
- **Better memory efficiency** - uses unified memory
- **Native Apple Silicon support** - no CUDA/BitsAndBytes compatibility issues
- **Smaller models** - Efficient 4-bit quantized models available

---

## Prerequisites

You're in: `/Users/yh/Downloads/FYP/20Jan/cf-armor` with `.venv` activated

**Check MLX installation:**
```bash
python -c "import mlx.core as mx; print(f'MLX {mx.__version__}')"
```

If not installed:
```bash
pip install mlx mlx-lm
```

---

## Quick Start (3 Steps)

### 1. Process Data (if not done already)

```bash
python training/process_data.py
```

This creates Hugging Face format datasets. The MLX script will auto-convert them.

### 2. Train CF-ARMOR Model

```bash
python training/train_mlx.py --dataset hybrid --iters 600
```

**Parameters:**
- `--dataset hybrid` - Main CF-ARMOR model (recommended)
- `--iters 600` - Number of training iterations (600 is ~3 epochs for 100 examples)
- `--batch-size 4` - Default batch size (can increase to 8 on M1 Pro/Max/Ultra)
- `--lr 1e-5` - Learning rate

**Expected time on M1:**
- M1 Base (8GB): ~15-20 minutes
- M1 Pro/Max (16-32GB): ~10-15 minutes
- M1 Ultra (64GB+): ~8-10 minutes

### 3. Test the Model

```bash
# Interactive testing
python training/test_mlx.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_YYYYMMDD_HHMMSS \
    --interactive

# Or test with predefined samples
python training/test_mlx.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_YYYYMMDD_HHMMSS \
    --test-samples
```

---

## What Happens During Training

1. **Data Conversion** (automatic)
   - Converts Hugging Face format → MLX format (JSONL with 'text' field)
   - Applies Qwen chat template
   - Saves to `data/mlx/{dataset}/`

2. **Model Loading**
   - Downloads `mlx-community/Qwen2.5-3B-Instruct-4bit` (~2GB)
   - 4-bit quantized for M1 efficiency

3. **LoRA Training**
   - Trains low-rank adapters (default rank=16)
   - Saves checkpoints every 100 iterations
   - Final adapter saved to `outputs/models_mlx/`

4. **Output**
   - LoRA adapter files (~50MB)
   - Training logs
   - `training_info.json` with config

---

## Training Options

### Train All 4 Models for Comparison

```bash
# 1. Direct rejection baseline
python training/train_mlx.py --dataset direct --iters 600

# 2. Intent extraction (ARMOR-style)
python training/train_mlx.py --dataset intent --iters 600

# 3. Consequence simulation
python training/train_mlx.py --dataset consequence --iters 600

# 4. Hybrid (CF-ARMOR - main contribution)
python training/train_mlx.py --dataset hybrid --iters 600
```

### Adjust Training Speed/Quality

**Faster training (lower quality):**
```bash
python training/train_mlx.py --dataset hybrid --iters 300 --batch-size 8
```

**Higher quality (slower):**
```bash
python training/train_mlx.py --dataset hybrid --iters 1200 --batch-size 2 --lr 5e-6
```

**More LoRA parameters (better but slower):**
```bash
python training/train_mlx.py --dataset hybrid --iters 600 --lora-rank 32 --lora-alpha 64
```

---

## Available Models

MLX community has many quantized models optimized for Apple Silicon:

### Qwen Models (Recommended)
```bash
# 3B model (default, fast)
--model mlx-community/Qwen2.5-3B-Instruct-4bit

# 7B model (better quality, slower)
--model mlx-community/Qwen2.5-7B-Instruct-4bit

# 14B model (best quality, M1 Pro/Max/Ultra only)
--model mlx-community/Qwen2.5-14B-Instruct-4bit
```

### Other Options
```bash
# Phi-3 (what you used before)
--model mlx-community/Phi-3-mini-4k-instruct-4bit

# Llama 3
--model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Mistral
--model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

---

## Memory Requirements

| Model | M1 8GB | M1 Pro 16GB | M1 Max 32GB | M1 Ultra 64GB |
|-------|--------|-------------|-------------|---------------|
| 3B    | ✅      | ✅           | ✅           | ✅             |
| 7B    | ⚠️ Tight | ✅         | ✅           | ✅             |
| 14B   | ❌      | ⚠️ Tight    | ✅           | ✅             |

---

## Troubleshooting

### "Out of memory"
**Solution:** Reduce batch size or use smaller model
```bash
python training/train_mlx.py --dataset hybrid --batch-size 2 --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

### "Model not found"
**Solution:** The model will auto-download from Hugging Face. Check internet connection.

### "No module named 'mlx'"
**Solution:** Install MLX
```bash
pip install mlx mlx-lm
```

### Training is slow
**Check:** Make sure you're using MLX models (with `mlx-community/` prefix)
**Check:** Close other apps to free up unified memory
**Check:** Use Activity Monitor to verify Python is using GPU

### Want to use original PyTorch script on M1
Not recommended, but if needed, here are the changes required:

1. Remove BitsAndBytes quantization
2. Change device to MPS:
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```
3. Use fp16 instead of bf16
4. Reduce batch size to 1

**But seriously, just use MLX - it's much faster!**

---

## Expected Output Locations

After training:

```
outputs/
├── models_mlx/
│   └── hybrid_20260121_HHMMSS/
│       ├── adapters.safetensors    ← Your trained LoRA weights
│       ├── adapter_config.json
│       └── training_info.json
└── data/
    └── mlx/
        └── hybrid/
            ├── train.jsonl         ← Auto-converted training data
            └── valid.jsonl         ← Auto-converted validation data
```

---

## Performance Comparison

Training 100 examples for 600 iterations on hybrid dataset:

| Framework | Device | Time | Memory |
|-----------|--------|------|--------|
| **MLX** | M1 | ~15 min | ~6GB |
| PyTorch | M1 (MPS) | ~45 min | ~8GB |
| PyTorch | NVIDIA 3090 | ~20 min | ~12GB |

**MLX is 3x faster than PyTorch on M1!**

---

## Full Command Reference

```bash
# 1. Process data (one-time)
python training/process_data.py

# 2. Train with MLX (recommended for M1)
python training/train_mlx.py --dataset hybrid --iters 600

# 3. Test the model
python training/test_mlx.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_*/final \
    --interactive

# Alternative: Train with PyTorch (slower, not recommended)
# python training/train.py --dataset hybrid --epochs 3
```

---

## Monitoring Training

Training will show:
```
Iter 100: Train loss 1.234, Val loss 1.456, It/sec 2.3
Iter 200: Train loss 0.987, Val loss 1.123, It/sec 2.4
Iter 300: Train loss 0.765, Val loss 0.998, It/sec 2.3
...
```

**Good signs:**
- ✅ Train loss decreasing (starts ~2.0, drops to ~0.5)
- ✅ Val loss decreasing
- ✅ It/sec > 2.0 (iterations per second)
- ✅ No memory errors

---

## Next Steps After Training

1. **Test the model** with harmful and benign prompts
2. **Evaluate** on jailbreak test sets
3. **Compare** with baseline models
4. **Tune hyperparameters** if needed (more iters, higher rank)

---

Done! 🚀
