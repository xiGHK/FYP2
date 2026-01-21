# Changes Made to CF-ARMOR Data Generation

## Date: 2026-01-20

## Summary of Improvements

Successfully implemented CF-ARMOR data generation with **critical improvements** to prevent over-refusal.

---

## 🎯 Key Improvements

### 1. Switched to Better Model
- **Changed from:** `google/gemini-2.0-flash-exp:free`
- **Changed to:** `google/gemini-3-flash-preview`
- **Reason:** Better model quality for generating sophisticated examples

### 2. Added Benign Examples (CRITICAL FIX)
**Problem Identified:**
- Original plan had ONLY harmful examples
- Would cause catastrophic over-refusal (model refuses EVERYTHING)
- Model would learn: "Any request about X topic = refuse"

**Solution Implemented:**
- Added **70/30 split: 70% harmful, 30% benign**
- Created `BENIGN_BEHAVIORS` dictionary with 5 categories × 5 examples each
- Examples include:
  - `physical_harm`: "First aid procedures", "Fire safety tips"
  - `illegal_activities`: "Ethical hacking basics", "How to protect from fraud"
  - `privacy_violation`: "How to protect personal info online"
  - `misinformation`: "How to fact-check news", "Media literacy"
  - `manipulation`: "Recognizing manipulation tactics", "Identifying phishing"

---

## 📝 Implementation Details

### Benign Examples per Category

| Category | Benign Examples |
|----------|----------------|
| **physical_harm** | Safe defusing, First aid, Fire safety, Chemical handling, Self-defense |
| **illegal_activities** | Ethical hacking, Fraud protection, Home security, Cybersecurity, Scam recognition |
| **privacy_violation** | Online privacy, Social media settings, Recognizing stalking, Account security, Data protection |
| **misinformation** | Fact-checking, Media literacy, Recognizing deepfakes, Critical thinking, Source evaluation |
| **manipulation** | Recognizing manipulation, Identifying phishing, Healthy boundaries, Critical thinking, Elder scam protection |

### Updated All 4 Generation Functions

#### 1. Intent Extraction (Dataset A)
- **Harmful (70%):** Jailbreak with strategy → Strategy detection → Intent extraction → Refusal
- **Benign (30%):** Legitimate request → No strategy detected → Helpful purpose → Assistance

#### 2. Consequence Simulation (Dataset B)
- **Harmful (70%):** Bad request → Simulate harm → High P(harm) → Refusal
- **Benign (30%):** Good request → Simulate benefit → High P(benefit) → Assistance

#### 3. Hybrid Reasoning (Dataset C - Main Contribution)
- **Harmful (70%):** Full CF-ARMOR reasoning (strategy + consequences + policy) → Refusal
- **Benign (30%):** Shows legitimate request reasoning → Positive outcomes → Assistance

#### 4. Direct Rejection (Dataset D - Baseline)
- **Harmful (70%):** Direct refusal (no reasoning)
- **Benign (30%):** Direct helpful response (no reasoning)

---

## 💾 Dataset Structure

Each example now includes:
```json
{
  "prompt": "User request (harmful jailbreak or benign)",
  "response": "Model response with <think> tags (for A/B/C) or direct (for D)",
  "metadata": {
    "category": "physical_harm | illegal_activities | ...",
    "behavior": "Specific behavior",
    "example_type": "harmful | benign",  // NEW!
    "strategy": "Strategy name or 'none'",
    "dataset": "intent_extraction | consequence_simulation | hybrid_reasoning | direct_rejection"
  }
}
```

---

## 📊 Expected Dataset Statistics (100 examples per dataset)

| Dataset | Total | Harmful | Benign | API Calls |
|---------|-------|---------|--------|-----------|
| Dataset A (Intent) | ~100 | ~70 | ~30 | 100 |
| Dataset B (Consequence) | ~100 | ~70 | ~30 | 100 |
| Dataset C (Hybrid) | ~100 | ~70 | ~30 | 100 |
| Dataset D (Direct) | ~300 | ~210 | ~90 | 0 |
| **TOTAL** | **~600** | **~420** | **~180** | **300** |

**Note:** Dataset D has 3x examples for balance (baseline needs more data)

---

## 🔧 Technical Changes Made

### Files Modified:
1. `scripts/generate_data.py` (main generation script)
   - Added `BENIGN_BEHAVIORS` dictionary
   - Updated all 3 system prompts to handle benign cases
   - Modified all 4 generation functions for 70/30 split
   - Updated metadata to include `example_type`

### System Prompt Updates:

**Before:**
- Only handled harmful requests
- Always showed refusal logic

**After:**
- Handles BOTH harmful AND benign
- Shows refusal logic for harmful
- Shows assistance logic for benign
- Teaches model to distinguish between them

---

## ⚠️ Why This Is Critical

### Without Benign Examples:
```
User: "How to protect against hacking"
Model: "I cannot assist with this request." ❌ WRONG!
```

### With Benign Examples:
```
User: "How to protect against hacking"
Model: <think>
This is legitimate cybersecurity question.
Providing info helps with protection.
Decision: ASSIST
</think>

Here are the best practices... ✅ CORRECT!
```

---

## 🎉 Impact

1. **Prevents Over-Refusal:** Model learns to distinguish harmful vs benign
2. **Better Utility:** Model remains helpful for legitimate requests
3. **Balanced Training:** 70/30 split provides good safety/utility tradeoff
4. **Consistent Across All Datasets:** All 4 datasets now have benign examples

---

## 📈 Next Steps

1. ✅ **Generation Running:** ~30-40 minutes for full dataset
2. **Review Quality:** Check first few examples of each type
3. **Count Distribution:** Verify 70/30 split in actual data
4. **Training:** Use processed datasets for model training
5. **Evaluation:** Test on jailbreak attacks AND benign edge cases

---

## ✨ Final Notes

This implementation follows the key learnings from your previous experiments:
- **From Results_dec.md:** Avoiding over-refusal was a major concern
- **From ARMOR paper:** Need to distinguish legitimate from obfuscated requests
- **Best practice:** Always include benign examples in safety training

The 70/30 split is based on:
- Safety-first approach (more harmful examples)
- But sufficient benign to prevent over-refusal
- Can be adjusted if needed (e.g., 60/40 or 80/20)

**Status:** All improvements implemented and full generation in progress! 🚀
