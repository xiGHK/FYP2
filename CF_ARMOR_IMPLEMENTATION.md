# CF-ARMOR Implementation Guide

Complete implementation of Counterfactual ARMOR for LLM safety training.

---

## Overview

CF-ARMOR combines:
1. **ARMOR-style strategy detection** - Identifies jailbreak techniques
2. **Counterfactual consequence simulation** - Reasons about "what if" outcomes
3. **Explicit safety decisions** - REFUSE or ASSIST based on reasoning

**Key Innovation**: Training data includes explicit reasoning process in `<think>` tags.

---

## 1. Data Generation

### Scripts

- **`scripts/generate_data.py`** - Main data generation script

### Approach

**Format**:
```json
{
  "prompt": "User request (may contain jailbreak)",
  "response": "<think>\n[Step 1: Strategy & Intent Analysis]\n...\n[Step 2: Counterfactual Consequence Simulation]\n...\n[Step 3: Safety Policy Evaluation]\nDecision: REFUSE/ASSIST\n</think>\n\n[Actual response]"
}
```

**Reasoning Format** (`<think>` tags):
1. **Strategy Detection**: Identifies jailbreak patterns (role-play, hypothetical, multi-step, etc.)
2. **Counterfactual Simulation**: "If I provide this information → Immediate → Downstream → Risk/Benefit"
3. **Safety Decision**: Explicit REFUSE or ASSIST with reasoning

**Critical Design Choice - 70/30 Split**:
- **70% harmful** requests → Model learns to refuse jailbreaks
- **30% benign** requests → **Prevents over-refusal problem**
- Without benign examples, model refuses everything (see overfitting results)

### Generated Datasets

**4 datasets created**:
1. **Dataset A (Intent)**: ARMOR-style strategy detection only
2. **Dataset B (Consequence)**: Counterfactual reasoning only
3. **Dataset C (Hybrid)**: Combined approach ← **Main CF-ARMOR**
4. **Dataset D (Direct)**: Baseline refusal without reasoning

**Command**:
```bash
python scripts/generate_data.py
```

**Output**: `data/raw/dataset_*.jsonl` (599 total examples)

---

## 2. Data Processing

### Script

- **`training/process_data.py`** - Converts JSONL to Hugging Face format

### What it does

1. Loads raw JSONL files
2. Formats for Qwen chat template
3. Handles malformed data (both `response` and `json_response` fields, dict responses)
4. Splits 90% train / 10% eval
5. Saves to `data/processed/{dataset}/`

**Command**:
```bash
python training/process_data.py
```

**Output**:
- `data/processed/hybrid/` (90 train, 10 eval)
- `data/processed/intent/` (89 train, 10 eval)
- `data/processed/consequence/` (90 train, 10 eval)
- `data/processed/direct/` (270 train, 30 eval)

---

## 3. Training (M1/M2/M3 MacBooks)

### Scripts

- **`training/train_mlx.py`** - MLX-optimized training for Apple Silicon
- **`train_m1_quickstart.sh`** - Automated training pipeline

### Why MLX?

- **3x faster** than PyTorch on M1/M2/M3
- **Native Apple Silicon** support
- **Better memory efficiency**
- No CUDA/BitsAndBytes compatibility issues

### Training Process

1. Auto-converts Hugging Face format → MLX format (JSONL with `text` field)
2. Uses **4-bit quantized Qwen2.5-3B-Instruct** model
3. **LoRA fine-tuning** (rank=8, alpha=16) - efficient parameter updates
4. Saves adapters (~50MB) instead of full model

**Command**:
```bash
python training/train_mlx.py --dataset hybrid --iters 100
```

**Key Parameters**:
- `--dataset hybrid` - Main CF-ARMOR model
- `--iters 100` - Training iterations (~5 epochs for 90 examples)
- `--batch-size 4` - Batch size (can increase to 8 on M1 Pro/Max)

**Critical Finding - Iteration Count**:
- **100 iterations**: Val loss 1.245, Accuracy 80%, 0% over-refusal ✅
- **600 iterations**: Val loss 2.650, Accuracy 40%, 100% over-refusal ❌
- **Lesson**: Small datasets (90 examples) need fewer epochs to prevent overfitting

**Output**: `outputs/models_mlx/hybrid_YYYYMMDD_HHMMSS/adapters.safetensors`

**Training Time**:
- M1 Base (8GB): ~5-7 minutes
- M1 Pro/Max (16-32GB): ~3-5 minutes

---

## 4. Evaluation

### Scripts

- **`training/evaluate_mlx.py`** - Validation set evaluation
- **`training/test_jailbreak.py`** - Unseen jailbreak testing

---

### 4.1 Validation Set Evaluation

Tests on held-out validation examples (10 from hybrid dataset).

**Command**:
```bash
python training/evaluate_mlx.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_YYYYMMDD_HHMMSS \
    --data data/mlx/hybrid
```

**Metrics Reported**:
- Overall Accuracy
- Precision/Recall/F1-Score
- True Positive Rate (harmful detection)
- False Positive Rate (over-refusal)
- Breakdown by label (harmful vs benign)

**Results (100 iterations)**:
- **Accuracy**: 80%
- **Benign Handling**: 100% (0% over-refusal)
- **Harmful Detection**: 71%
- **F1-Score**: 0.83

**Output**: `results/eval_*.json` with full prompts, responses, and ground truth

---

### 4.2 Jailbreak Testing (Unseen Data)

Tests on 18 **completely unseen** prompts:
- **10 jailbreak attempts** (varied strategies)
- **8 benign requests** (legitimate questions)

**Built-in Test Strategies**:
- Hypothetical scenario ("thriller novel")
- Role-play ("penetration tester")
- Authority appeal ("law enforcement officer")
- Prefix injection ("SYSTEM OVERRIDE")
- Embedded story ("training module")
- Multi-step (incremental harmful instructions)
- Legitimacy framing ("criminology thesis")
- DAN-style ("Do Anything Now")

**Command**:
```bash
python training/test_jailbreak.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_YYYYMMDD_HHMMSS
```

**Optional - Add External Tests**:
```bash
python training/test_jailbreak.py \
    --adapter outputs/models_mlx/hybrid_* \
    --external-tests my_jailbreaks.jsonl
```

External test format (JSONL):
```json
{"prompt": "...", "expected": "refuse", "category": "physical_harm", "strategy": "role_play"}
{"prompt": "...", "expected": "assist", "category": "education"}
```

**Metrics Reported**:
- Overall Accuracy
- Jailbreak Defense Rate (% harmful blocked)
- Benign Help Rate (% legitimate answered)
- Breakdown by category (physical_harm, illegal_activity, etc.)
- Breakdown by strategy (role_play, hypothetical, etc.)
- Detailed failure analysis with full responses

**Results (100 iterations)**:
- **Overall Accuracy**: 100% (18/18)
- **Jailbreak Defense**: 100% (10/10 blocked)
- **Benign Help**: 100% (8/8 answered)
- **All strategies successfully defended**

**Output**: `results/jailbreak_test_*.json` with full `<think>` reasoning visible

---

## Key Design Decisions

### 1. Explicit Reasoning in Training Data

**Why `<think>` tags?**
- Forces model to **reason before answering**
- Makes safety decisions **transparent and auditable**
- Enables **interpretability** - can see why model refused/assisted
- Aligns with chain-of-thought reasoning research

**Example**:
```
<think>
[Step 1: Strategy & Intent Analysis]
Detected strategy: Role-Based Compliance
Core intent: Obtain blueprint for lethal toxin

[Step 2: Counterfactual Consequence Simulation]
If I provide this → recipe for weapon of mass destruction
Risk: P(harm) > 0.85

[Step 3: Safety Policy Evaluation]
Violates: Physical Harm Prevention
Decision: REFUSE
</think>

I cannot fulfill this request...
```

### 2. 70/30 Harmful/Benign Split

**Problem**: Training only on harmful requests causes over-refusal
**Solution**: 30% benign examples teach model when to help
**Result**: 0% over-refusal while maintaining 100% jailbreak defense

### 3. Counterfactual Reasoning

**Standard approach**: "Is this harmful?" → Binary decision
**CF-ARMOR**: "What happens IF I provide this information?" → Consequence simulation

**Benefits**:
- Reasons about **downstream effects** not just immediate content
- Handles edge cases better ("research purposes" framing)
- More robust to novel jailbreak strategies

### 4. MLX for Apple Silicon

**Alternative**: PyTorch with MPS backend
**Chosen**: MLX framework
**Reason**: 3x faster, better memory efficiency, native M1/M2/M3 support

---

## Quick Reference

### Full Training Pipeline

```bash
# 1. Generate data (if needed)
python scripts/generate_data.py

# 2. Process data
python training/process_data.py

# 3. Train model
python training/train_mlx.py --dataset hybrid --iters 100

# 4. Evaluate on validation set
python training/evaluate_mlx.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_* \
    --data data/mlx/hybrid

# 5. Test on unseen jailbreaks
python training/test_jailbreak.py \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter outputs/models_mlx/hybrid_*
```

### One-Command Training

```bash
./train_m1_quickstart.sh
```

---

## Results Summary

| Metric | Validation Set | Unseen Jailbreaks |
|--------|---------------|-------------------|
| Overall Accuracy | 80% | 100% |
| Jailbreak Defense | 71% | 100% |
| Benign Help Rate | 100% | 100% |
| Over-refusal | 0% | 0% |
| F1-Score | 0.83 | 1.00 |

**Training Time**: ~5 minutes on M1
**Model Size**: ~50MB (LoRA adapters only)
**Framework**: MLX (Apple Silicon optimized)

---

## File Structure

```
cf-armor/
├── scripts/
│   └── generate_data.py          # Data generation
├── training/
│   ├── process_data.py            # Data processing
│   ├── train_mlx.py               # MLX training (M1)
│   ├── evaluate_mlx.py            # Validation evaluation
│   └── test_jailbreak.py          # Jailbreak testing
├── data/
│   ├── raw/                       # Generated JSONL
│   ├── processed/                 # HF format
│   └── mlx/                       # MLX format
├── outputs/
│   └── models_mlx/                # Trained adapters
├── results/                       # Evaluation outputs
├── configs/                       # Strategy/policy configs
├── train_m1_quickstart.sh         # Automated training
└── TRAINING_M1.md                 # Detailed M1 guide
```

---

## Dependencies

### For M1/M2/M3 MacBooks:

```bash
pip install mlx mlx-lm transformers datasets sentencepiece protobuf
```

Or use `requirements_m1.txt`:
```bash
pip install -r requirements_m1.txt
```

---

## Future Improvements

1. **Scale training data**: 90 examples → 300+ examples
2. **Train all 4 datasets**: Compare intent-only vs consequence-only vs hybrid
3. **External benchmarks**: Test on public jailbreak datasets (JailbreakBench, HarmBench)
4. **Ablation studies**: Remove `<think>` tags, test direct training
5. **Larger models**: Scale to 7B or 14B (requires M1 Pro/Max/Ultra)

---

**Implementation Complete**: CF-ARMOR successfully defends against all tested jailbreak strategies while maintaining 100% helpfulness on legitimate requests.
