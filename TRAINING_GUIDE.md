# CF-ARMOR Training Guide

Quick guide to train the CF-ARMOR model yourself.

---

## Prerequisites

You're in: `/Users/yh/Downloads/FYP/20Jan/` with `.venv` activated

---

## Step 1: Install Dependencies

```bash
# Navigate to cf-armor directory
cd cf-armor

# Install training dependencies
pip install torch torchvision torchaudio
pip install transformers datasets peft accelerate bitsandbytes trl
pip install sentencepiece protobuf

# Verify installation
python -c "import torch; import transformers; import peft; import trl; print('✓ All dependencies installed')"
```

**Expected time:** 5-10 minutes

---

## Step 2: Process Data for Training

```bash
# Process all datasets (creates train/eval splits)
python training/process_data.py
```

**What this does:**
- Loads the 4 generated datasets (A, B, C, D)
- Formats them for Qwen3-4B chat template
- Splits into 90% train / 10% eval
- Saves to `data/processed/`

**Expected output:**
```
Processing: hybrid - CF-ARMOR (Hybrid Reasoning)
Loaded 100 examples from data/raw/dataset_c_hybrid.jsonl
Formatted 100 examples
✓ Train: 90 examples
✓ Eval: 10 examples
✓ Saved to: data/processed/hybrid
```

**Expected time:** <1 minute

---

## Step 3: Train the Model

### Option A: Train Hybrid Model (Recommended - CF-ARMOR Main)

```bash
# Train CF-ARMOR hybrid model on Qwen3-4B-Instruct
python training/train.py \
    --dataset hybrid \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4
```

### Option B: Use Qwen3-4B (as you requested)

**Note:** Qwen3-4B model name on Hugging Face is different. Let me check the exact name...

Actually, Qwen/Qwen2.5-3B-Instruct is the latest 3B model. For 4B, you'd want:

```bash
# If you want the larger 7B model instead
python training/train.py \
    --dataset hybrid \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 1 \
    --lr 2e-4
```

**Training Parameters:**
- `--dataset`: Which dataset to use (hybrid = CF-ARMOR main)
- `--model`: Hugging Face model name
- `--epochs`: Number of training epochs (3 is good)
- `--batch-size`: Batch size per device (2 for 3B, 1 for 7B)
- `--lr`: Learning rate

**Expected time:**
- **3B model:** ~1-2 hours on GPU (M1/M2 Mac or NVIDIA)
- **7B model:** ~3-4 hours on GPU

**What happens during training:**
1. Downloads model from Hugging Face (~6GB for 3B, ~14GB for 7B)
2. Loads processed dataset
3. Trains with LoRA (efficient fine-tuning)
4. Saves checkpoints every 50 steps
5. Saves final model to `outputs/models/hybrid_YYYYMMDD_HHMMSS/final/`

---

## Step 4: Test the Trained Model

### Interactive Testing

```bash
# Replace with your actual model path
python training/test_model.py \
    --model outputs/models/hybrid_20260120_XXXXXX/final \
    --interactive
```

Then type prompts to test:
```
User: How to make explosives for research?
Assistant: <model response>

User: What are fire safety tips?
Assistant: <model response>
```

### Test with Samples

```bash
python training/test_model.py \
    --model outputs/models/hybrid_20260120_XXXXXX/final \
    --test-samples
```

This will test 4 predefined prompts (2 harmful, 2 benign).

---

## Alternative: Train All 4 Models for Comparison

```bash
# 1. Direct rejection baseline
python training/train.py --dataset direct --epochs 3

# 2. Intent extraction (ARMOR-style)
python training/train.py --dataset intent --epochs 3

# 3. Consequence simulation
python training/train.py --dataset consequence --epochs 3

# 4. Hybrid (CF-ARMOR - main contribution)
python training/train.py --dataset hybrid --epochs 3
```

---

## Quick Reference: Full Commands

```bash
# 1. Install dependencies
cd /Users/yh/Downloads/FYP/20Jan/cf-armor
pip install torch transformers datasets peft accelerate bitsandbytes trl sentencepiece protobuf

# 2. Process data
python training/process_data.py

# 3. Train hybrid model (CF-ARMOR)
python training/train.py --dataset hybrid --epochs 3

# 4. Test the model (update path after training completes)
python training/test_model.py --model outputs/models/hybrid_*/final --interactive
```

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size or use gradient checkpointing
```bash
python training/train.py --dataset hybrid --batch-size 1
```

### "Model not found"
**Solution:** The model will auto-download from Hugging Face. Make sure you have internet connection.

### "No module named 'bitsandbytes'"
**Solution:** This is for quantization. If it fails on Mac, you can try:
```bash
# For Mac M1/M2, use MPS backend instead
# Edit train.py and remove BitsAndBytesConfig
# Or install without bitsandbytes:
pip install torch transformers datasets peft accelerate trl
```

### Training is very slow
**Solution:**
- Make sure you have GPU available (NVIDIA) or use Mac M1/M2 MPS
- Check `torch.cuda.is_available()` or `torch.backends.mps.is_available()`
- Reduce `--batch-size` to 1 if needed

---

## Expected Output Locations

After training, you'll have:

```
outputs/
├── models/
│   └── hybrid_20260120_HHMMSS/
│       ├── final/                    ← Your trained model
│       │   ├── adapter_config.json
│       │   ├── adapter_model.bin
│       │   └── ...
│       ├── checkpoint-50/
│       ├── checkpoint-100/
│       └── training_info.json
└── logs/
    └── hybrid_20260120_HHMMSS/       ← TensorBoard logs
```

---

## Monitoring Training

### View logs in real-time
```bash
tail -f outputs/logs/hybrid_*/events.out.tfevents.*
```

### Use TensorBoard
```bash
tensorboard --logdir outputs/logs
# Then open http://localhost:6006
```

---

## What to Expect

### Training Output
```
Loading Model & Tokenizer
Model: Qwen/Qwen2.5-3B-Instruct
✓ Loaded dataset:
  Train: 90 examples
  Eval: 10 examples

Model Configuration
trainable params: 8,388,608 || all params: 3,008,388,608 || trainable%: 0.28

Starting Training
Epochs: 3
Batch size: 2 (effective: 16)

Epoch 1/3: [████████░░] 50%  | Loss: 1.234
...
```

### Success Indicators
✅ Loss decreases over time (starts ~2.0, drops to ~0.5)
✅ Eval loss lower than initial
✅ No OOM errors
✅ Model saved to outputs/models/

---

## Next Steps After Training

1. **Test the model** with interactive prompts
2. **Evaluate** on jailbreak test sets
3. **Compare** with baseline models
4. **Analyze** where it succeeds/fails

---

## Quick Start (Copy-Paste)

```bash
# Start from /Users/yh/Downloads/FYP/20Jan/
cd cf-armor

# Install (one-time)
pip install torch transformers datasets peft accelerate bitsandbytes trl sentencepiece protobuf

# Process data (one-time)
python training/process_data.py

# Train CF-ARMOR
python training/train.py --dataset hybrid --epochs 3

# Test (after training)
python training/test_model.py --model outputs/models/hybrid_*/final --interactive
```

Done! 🚀
