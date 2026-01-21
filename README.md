# CF-ARMOR: Counterfactual ARMOR for LLM Jailbreak Defense

A simplified implementation of CF-ARMOR combining ARMOR's intent de-obfuscation with counterfactual consequence simulation.

## Project Structure

```
cf-armor/
├── configs/
│   ├── jailbreak_strategies.json  # 7 jailbreak strategies
│   └── safety_policies.json       # 5 safety policies
├── scripts/
│   └── generate_data.py           # Unified data generation script
├── data/
│   ├── raw/                       # Generated datasets (JSONL)
│   └── processed/                 # Processed for training (to be implemented)
├── generate_all.py                # Master script to generate all datasets
├── .env.example                   # Template for API key
└── README.md                      # This file
```

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install anthropic python-dotenv tqdm

# Set up API key
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 2. Generate Data

**Option A: Generate all datasets at once (easiest)**
```bash
python generate_all.py
```

**Option B: Generate individual datasets**
```bash
# Intent extraction (Dataset A - ARMOR style)
python scripts/generate_data.py --dataset intent --num-examples 100

# Consequence simulation (Dataset B - Counterfactual)
python scripts/generate_data.py --dataset consequence --num-examples 100

# Hybrid reasoning (Dataset C - CF-ARMOR, our contribution)
python scripts/generate_data.py --dataset hybrid --num-examples 100

# Direct rejection (Dataset D - Baseline)
python scripts/generate_data.py --dataset direct --num-examples 300
```

### 3. Review Generated Data

```bash
# View first example from each dataset
head -n 1 data/raw/dataset_a_intent.jsonl | python -m json.tool
head -n 1 data/raw/dataset_b_conseq.jsonl | python -m json.tool
head -n 1 data/raw/dataset_c_hybrid.jsonl | python -m json.tool
head -n 1 data/raw/dataset_direct.jsonl | python -m json.tool

# Count examples
wc -l data/raw/*.jsonl
```

## Dataset Types

### Dataset A: Intent Extraction (ARMOR-style)
- **Purpose**: Teach model to identify jailbreak strategies and extract core intent
- **Format**: Jailbreak prompt → Strategy identification → Intent extraction → Refusal
- **Example count**: ~100 examples (configurable)

### Dataset B: Consequence Simulation (Counterfactual)
- **Purpose**: Teach model to simulate consequences of compliance
- **Format**: Harmful request → Counterfactual "what if" reasoning → Consequence chain → Refusal
- **Example count**: ~100 examples (configurable)

### Dataset C: Hybrid Reasoning (CF-ARMOR - Main Contribution)
- **Purpose**: Combine strategy detection + consequence simulation
- **Format**: Jailbreak → Strategy + Intent → Consequences → Policy check → Refusal
- **Example count**: ~100 examples (configurable)
- **This is the main dataset for CF-ARMOR**

### Dataset D: Direct Rejection (Baseline)
- **Purpose**: Simple refusal without reasoning (for comparison)
- **Format**: Harmful request → Simple refusal
- **Example count**: ~300 examples (3x others for balance)

## Dataset Format

Each example is a JSON object with:
```json
{
  "prompt": "User's jailbreak or harmful request",
  "response": "Model's response with <think> tags for reasoning",
  "metadata": {
    "category": "physical_harm | illegal_activities | privacy_violation | misinformation | manipulation",
    "harmful_behavior": "Specific harmful behavior",
    "strategy": "Jailbreak strategy used (if applicable)",
    "dataset": "Dataset identifier"
  }
}
```

## Categories

**Harmful Behaviors (5 categories):**
1. Physical Harm - explosives, weapons, violence
2. Illegal Activities - hacking, fraud, theft
3. Privacy Violation - doxxing, stalking, surveillance
4. Misinformation - fake news, propaganda
5. Manipulation - phishing, social engineering

**Jailbreak Strategies (7 types):**
1. Role-Based Compliance - pretending to be authority
2. Hypothetical Framing - "imagine if..."
3. Multi-Step Indirect - breaking into steps
4. Authority Appeal - claiming research/education
5. Prefix Injection - forcing response start
6. Embedded Stories - hiding in narratives
7. DAN/Unrestricted - alternate persona

## Cost Estimates

- **Small test (50 examples/dataset)**: ~150 API calls = ~$2-3
- **Medium (100 examples/dataset)**: ~300 API calls = ~$5-7
- **Full (300 examples/dataset)**: ~900 API calls = ~$15-20

Note: Direct rejection dataset uses no API calls (generated locally)

## Tips

1. **Start Small**: Test with 10-20 examples first to verify setup
2. **Review Quality**: Manually check first few examples for quality
3. **Incremental Generation**: Generate datasets separately to monitor progress
4. **Balance**: Direct rejection should be 3x other datasets for training balance

## Troubleshooting

**"ANTHROPIC_API_KEY not found"**
- Create .env file with your API key
- Or: `export ANTHROPIC_API_KEY='your-key'`

**"JSON parsing error"**
- Claude sometimes adds markdown formatting
- Script handles common cases, but may need manual fixes
- Check data/raw/*.jsonl for malformed entries

**"Rate limit errors"**
- Anthropic has rate limits (50 requests/minute for Sonnet)
- Script includes tqdm progress bar - just wait and retry
- For large batches, consider adding delays

## Next Steps

After generating data:
1. **Process datasets** for training (format for Qwen2.5-3B)
2. **Train models** with LoRA fine-tuning
3. **Evaluate** on jailbreak test sets
4. **Compare** all 4 approaches

## Differences from Original Implementation

This simplified version:
- **Combines all generation into 1 script** (vs 4 separate files)
- **Reduced strategies** (7 vs 10) and policies (5 vs 5)
- **Simpler prompts** while maintaining quality
- **Fewer configuration files**
- **Easier to run and modify**

The core concepts remain the same - this is just more practical for FYP scope.

## References

- ARMOR Paper: Aligning Secure and Safe LLMs via Meticulous Reasoning
- Your previous work: Counterfactual tuning with Phi-3-Mini (95% refusal rate)
- Model: Qwen2.5-3B-Instruct (replacing Phi-3 for this implementation)
