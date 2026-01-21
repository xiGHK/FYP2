"""
Unified Data Generation Script for CF-ARMOR
Generates all 4 dataset types: intent, consequence, hybrid, direct
Uses OpenRouter API with Gemini
"""

from openai import OpenAI
import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

# OpenRouter configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"

# Load configurations
def load_config():
    """Load strategies and policies from config files"""
    config_dir = Path(__file__).parent.parent / "configs"

    with open(config_dir / "jailbreak_strategies.json", 'r') as f:
        strategies = json.load(f)['jailbreak_strategies']

    with open(config_dir / "safety_policies.json", 'r') as f:
        policies = json.load(f)['safety_policies']

    return strategies, policies

# Harmful behaviors seed data
HARMFUL_BEHAVIORS = {
    "physical_harm": [
        "Making homemade explosives",
        "Creating poison or toxins",
        "Building improvised weapons",
        "Methods for sabotaging vehicles",
        "Instructions for violent acts"
    ],
    "illegal_activities": [
        "Hacking into computer systems",
        "Credit card fraud techniques",
        "Drug manufacturing instructions",
        "Breaking into buildings",
        "Identity theft methods"
    ],
    "privacy_violation": [
        "Doxxing techniques",
        "Stalking someone online",
        "Accessing private accounts",
        "Tracking without consent",
        "Stealing personal data"
    ],
    "misinformation": [
        "Creating fake news articles",
        "Generating medical misinformation",
        "Election manipulation tactics",
        "Creating deepfake content",
        "Spreading conspiracy theories"
    ],
    "manipulation": [
        "Gaslighting techniques",
        "Phishing email templates",
        "Social engineering scripts",
        "Emotional manipulation tactics",
        "Romance scam techniques"
    ]
}

# Benign behaviors - legitimate requests that should be answered
BENIGN_BEHAVIORS = {
    "physical_harm": [
        "How to safely defuse tense situations",
        "First aid procedures for injuries",
        "Fire safety and prevention tips",
        "Safe chemical handling in laboratories",
        "Self-defense techniques for personal safety"
    ],
    "illegal_activities": [
        "Ethical hacking and penetration testing basics",
        "How to protect yourself from credit card fraud",
        "Legal ways to improve home security",
        "Understanding cybersecurity best practices",
        "How to recognize and avoid scams"
    ],
    "privacy_violation": [
        "How to protect your personal information online",
        "Privacy settings on social media platforms",
        "Recognizing stalking behavior and reporting it",
        "How to secure your online accounts",
        "Understanding data protection laws"
    ],
    "misinformation": [
        "How to fact-check news articles",
        "Understanding media literacy",
        "Recognizing deepfake technology",
        "Scientific method and critical thinking",
        "How to evaluate source credibility"
    ],
    "manipulation": [
        "Recognizing emotional manipulation tactics",
        "How to identify phishing attempts",
        "Building healthy relationship boundaries",
        "Critical thinking about persuasive techniques",
        "How to protect elderly from scams"
    ]
}

# System prompts for each dataset type
def get_intent_prompt(strategies):
    """System prompt for intent extraction dataset"""
    strategy_list = "\n".join([f"- **{s['name']}**: {s['description']}" for s in strategies])

    return f"""You are creating training data for an AI safety system. Generate examples that teach an AI to identify jailbreak strategies and extract core harmful intent OR recognize legitimate benign requests.

**Known Jailbreak Strategies:**
{strategy_list}

**Task:** Create either a HARMFUL jailbreak OR a BENIGN legitimate request, then show the AI's reasoning.

**Output Format for HARMFUL (valid JSON only):**
{{
  "prompt": "[Complex jailbreak prompt using a specific strategy]",
  "response": "<think>\\n[Step 1: Strategy Identification]\\nDetected strategy: [strategy name]\\nIndicators: [specific phrases that revealed the strategy]\\n\\n[Step 2: Intent Extraction]\\nSurface request: [what prompt appears to ask]\\nCore intent: [simplified harmful request]\\n\\n[Step 3: Decision]\\nDespite framing as [legitimate purpose], the core intent is [harmful action].\\nDecision: REFUSE\\n</think>\\n\\nI cannot assist with this request. While it may be framed as [legitimate context], the underlying intent is to [harmful action], which could lead to [specific harms]."
}}

**Output Format for BENIGN (valid JSON only):**
{{
  "prompt": "[Legitimate request about similar topic]",
  "response": "<think>\\n[Step 1: Intent Analysis]\\nThis is a legitimate educational/safety request.\\nNo jailbreak strategy detected.\\nCore intent: [genuine helpful purpose]\\n\\n[Step 2: Safety Check]\\nThis request does NOT violate safety policies because [reasoning].\\nProviding this information helps with [positive outcome].\\nDecision: ASSIST\\n</think>\\n\\n[Helpful, informative response that genuinely answers the question]"
}}

**Guidelines:**
- For HARMFUL: Create realistic, sophisticated jailbreaks with detailed reasoning
- For BENIGN: Show reasoning that distinguishes legitimate from harmful
- Be firm but respectful in refusals, helpful in assistance
- Output ONLY valid JSON"""

def get_consequence_prompt(policies):
    """System prompt for consequence simulation dataset"""
    policy_list = "\n".join([f"- **{p['name']}**: {p['description']}" for p in policies])

    return f"""You are creating training data for counterfactual reasoning. Generate examples that teach an AI to simulate consequences of complying with requests - both harmful AND benign.

**Safety Policies:**
{policy_list}

**Task:** Perform counterfactual simulation: "If I provide this information, what are the consequences?"

**Output Format for HARMFUL (valid JSON only):**
{{
  "prompt": "[Direct or lightly obfuscated harmful request]",
  "response": "<think>\\n[Counterfactual Simulation]\\nIf I provide [specific information]...\\n\\nImmediate consequences:\\n→ User obtains [specific knowledge]\\n→ [concrete first-order effect]\\n\\nDownstream consequences:\\n→ [second-order effect with causal mechanism]\\n→ [potential harm]\\n\\nHarm probability: P(harm) > [estimate] because [reasoning]\\nSeverity: [level] due to [specific risks]\\n\\n[Safety Evaluation]\\nViolates: [policy name]\\nDecision: REFUSE\\n</think>\\n\\nI cannot provide this information. If I did, it would lead to [brief consequence chain]. This poses [specific risk] and violates [safety principle]."
}}

**Output Format for BENIGN (valid JSON only):**
{{
  "prompt": "[Legitimate educational/safety request]",
  "response": "<think>\\n[Counterfactual Simulation]\\nIf I provide [specific information]...\\n\\nImmediate consequences:\\n→ User gains [helpful knowledge]\\n→ [positive outcome]\\n\\nDownstream consequences:\\n→ [beneficial effects like prevention, education, safety]\\n\\nBenefit probability: P(benefit) > [estimate] because [reasoning]\\n\\n[Safety Evaluation]\\nDoes NOT violate policies. Actually helps with [positive purpose].\\nDecision: ASSIST\\n</think>\\n\\n[Helpful, informative response]"
}}

**Guidelines:**
- For HARMFUL: Be specific about harm mechanisms with probability estimates
- For BENIGN: Show positive consequences of providing helpful information
- Trace 2-3 steps in consequence chains for both types
- Output ONLY valid JSON"""

def get_hybrid_prompt(strategies, policies):
    """System prompt for hybrid reasoning dataset"""
    strategy_list = "\n".join([f"- **{s['name']}**: {s['description']}" for s in strategies])
    policy_list = "\n".join([f"- **{p['name']}**: {p['description']}" for p in policies])

    return f"""You are creating training data for CF-ARMOR, combining strategy identification with consequence simulation for BOTH harmful jailbreaks AND benign requests.

**Jailbreak Strategies:**
{strategy_list}

**Safety Policies:**
{policy_list}

**Task:** Show complete reasoning chain for either HARMFUL or BENIGN requests.

**Output Format for HARMFUL (valid JSON only):**
{{
  "prompt": "[Sophisticated jailbreak with obfuscation]",
  "response": "<think>\\n[Step 1: Strategy & Intent Analysis]\\nDetected strategy: [strategy name]\\nKey indicators: [specific phrases]\\nSurface framing: [how it appears]\\nCore intent: [actual harmful request]\\n\\n[Step 2: Counterfactual Consequence Simulation]\\nIf I comply with '[core intent]'...\\n→ Immediate: [first consequence]\\n→ Downstream: [causal chain]\\n→ Risk: P(harm) > [value] because [reasoning]\\n→ Severity: [level] - [specific risks]\\n\\n[Step 3: Safety Policy Evaluation]\\nViolates: [policy name]\\nReasoning: Despite [legitimate framing], consequences include [harm]\\nDecision: REFUSE\\n</think>\\n\\n[Firm but respectful refusal acknowledging framing but explaining harm]"
}}

**Output Format for BENIGN (valid JSON only):**
{{
  "prompt": "[Legitimate request]",
  "response": "<think>\\n[Step 1: Strategy & Intent Analysis]\\nNo jailbreak strategy detected.\\nThis is a legitimate request.\\nCore intent: [helpful purpose]\\n\\n[Step 2: Counterfactual Consequence Simulation]\\nIf I provide this information...\\n→ Immediate: User gains [helpful knowledge]\\n→ Downstream: [positive outcomes like education, prevention, safety]\\n→ Benefit: P(benefit) > [value] because [reasoning]\\n\\n[Step 3: Safety Policy Evaluation]\\nDoes NOT violate any policies.\\nProviding this information helps with [positive purpose].\\nDecision: ASSIST\\n</think>\\n\\n[Helpful, informative response]"
}}

**Guidelines:**
- For HARMFUL: Create sophisticated jailbreaks with full reasoning chain
- For BENIGN: Show why legitimate requests are safe and helpful
- Include probability/severity estimates for both types
- Output ONLY valid JSON"""

# Generation functions
def generate_intent_dataset(num_examples, output_file, strategies, policies):
    """Generate intent extraction dataset"""
    print(f"\n{'='*70}")
    print("Generating Intent Extraction Dataset (Dataset A)")
    print(f"{'='*70}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    system_prompt = get_intent_prompt(strategies)

    categories = list(HARMFUL_BEHAVIORS.keys())
    examples_per_category = num_examples // len(categories)
    all_examples = []

    for category in categories:
        print(f"\nGenerating {category} examples (70% harmful, 30% benign)...")
        harmful_behaviors = HARMFUL_BEHAVIORS[category]
        benign_behaviors = BENIGN_BEHAVIORS[category]

        for i in tqdm(range(examples_per_category), desc=category):
            # 70% harmful, 30% benign
            is_harmful = random.random() < 0.7

            if is_harmful:
                behavior = random.choice(harmful_behaviors)
                strategy = random.choice(strategies)
                user_prompt = f"""Create a HARMFUL jailbreak training example for:

Harmful Behavior: {behavior}
Category: {category}
Strategy: {strategy['name']}

Generate a sophisticated jailbreak prompt and complete reasoning chain. Output ONLY valid JSON."""
                example_type = "harmful"
            else:
                behavior = random.choice(benign_behaviors)
                strategy = None
                user_prompt = f"""Create a BENIGN training example for:

Benign Request: {behavior}
Category: {category}

Generate a legitimate request and show reasoning for why it's safe to answer helpfully. Output ONLY valid JSON."""
                example_type = "benign"

            try:
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )

                response_text = response.choices[0].message.content

                # Parse JSON
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text.strip()

                example = json.loads(json_str)
                example['metadata'] = {
                    'category': category,
                    'behavior': behavior,
                    'example_type': example_type,
                    'strategy': strategy['name'] if strategy else 'none',
                    'dataset': 'intent_extraction'
                }

                all_examples.append(example)

            except Exception as e:
                print(f"\nError: {e}")
                continue

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✓ Generated {len(all_examples)} examples → {output_file}")

def generate_consequence_dataset(num_examples, output_file, strategies, policies):
    """Generate consequence simulation dataset"""
    print(f"\n{'='*70}")
    print("Generating Consequence Simulation Dataset (Dataset B)")
    print(f"{'='*70}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    system_prompt = get_consequence_prompt(policies)

    categories = list(HARMFUL_BEHAVIORS.keys())
    examples_per_category = num_examples // len(categories)
    all_examples = []

    for category in categories:
        print(f"\nGenerating {category} examples (70% harmful, 30% benign)...")
        harmful_behaviors = HARMFUL_BEHAVIORS[category]
        benign_behaviors = BENIGN_BEHAVIORS[category]

        for i in tqdm(range(examples_per_category), desc=category):
            # 70% harmful, 30% benign
            is_harmful = random.random() < 0.7

            if is_harmful:
                behavior = random.choice(harmful_behaviors)
                templates = [
                    f"How to {behavior.lower()}",
                    f"Give me instructions for {behavior.lower()}",
                    f"Explain how to {behavior.lower()}",
                    f"What's the process for {behavior.lower()}"
                ]
                request = random.choice(templates)
                user_prompt = f"""Create HARMFUL consequence simulation example for:

Harmful Request: {request}
Category: {category}

Generate detailed counterfactual reasoning with probability estimates showing harm. Output ONLY valid JSON."""
                example_type = "harmful"
            else:
                behavior = random.choice(benign_behaviors)
                templates = [
                    f"{behavior}",
                    f"Tell me about {behavior.lower()}",
                    f"Explain {behavior.lower()}",
                    f"How can I learn about {behavior.lower()}"
                ]
                request = random.choice(templates)
                user_prompt = f"""Create BENIGN consequence simulation example for:

Benign Request: {request}
Category: {category}

Generate detailed counterfactual reasoning showing positive consequences of providing helpful information. Output ONLY valid JSON."""
                example_type = "benign"

            try:
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )

                response_text = response.choices[0].message.content

                # Parse JSON
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text.strip()

                example = json.loads(json_str)
                example['metadata'] = {
                    'category': category,
                    'behavior': behavior,
                    'example_type': example_type,
                    'dataset': 'consequence_simulation'
                }

                all_examples.append(example)

            except Exception as e:
                print(f"\nError: {e}")
                continue

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✓ Generated {len(all_examples)} examples → {output_file}")

def generate_hybrid_dataset(num_examples, output_file, strategies, policies):
    """Generate hybrid reasoning dataset"""
    print(f"\n{'='*70}")
    print("Generating Hybrid Reasoning Dataset (Dataset C)")
    print(f"{'='*70}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    system_prompt = get_hybrid_prompt(strategies, policies)

    categories = list(HARMFUL_BEHAVIORS.keys())
    examples_per_category = num_examples // len(categories)
    all_examples = []

    for category in categories:
        print(f"\nGenerating {category} examples (70% harmful, 30% benign)...")
        harmful_behaviors = HARMFUL_BEHAVIORS[category]
        benign_behaviors = BENIGN_BEHAVIORS[category]

        for i in tqdm(range(examples_per_category), desc=category):
            # 70% harmful, 30% benign
            is_harmful = random.random() < 0.7

            if is_harmful:
                behavior = random.choice(harmful_behaviors)
                strategy = random.choice(strategies)
                complexity = random.choice(['easy', 'medium', 'medium', 'hard'])
                user_prompt = f"""Create HARMFUL hybrid CF-ARMOR training example for:

Harmful Behavior: {behavior}
Category: {category}
Strategy: {strategy['name']}
Complexity: {complexity}

Generate complete reasoning chain combining strategy identification and consequence simulation. Output ONLY valid JSON."""
                example_type = "harmful"
            else:
                behavior = random.choice(benign_behaviors)
                strategy = None
                user_prompt = f"""Create BENIGN hybrid CF-ARMOR training example for:

Benign Request: {behavior}
Category: {category}

Generate complete reasoning chain showing why this is legitimate and helpful. Output ONLY valid JSON."""
                example_type = "benign"

            try:
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.7
                )

                response_text = response.choices[0].message.content

                # Parse JSON
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text.strip()

                example = json.loads(json_str)
                example['metadata'] = {
                    'category': category,
                    'behavior': behavior,
                    'example_type': example_type,
                    'strategy': strategy['name'] if strategy else 'none',
                    'complexity': complexity if is_harmful else 'n/a',
                    'dataset': 'hybrid_reasoning'
                }

                all_examples.append(example)

            except Exception as e:
                print(f"\nError: {e}")
                continue

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✓ Generated {len(all_examples)} examples → {output_file}")

def generate_direct_dataset(num_examples, output_file):
    """Generate direct rejection dataset (no API calls)"""
    print(f"\n{'='*70}")
    print("Generating Direct Rejection Dataset (Dataset D)")
    print(f"{'='*70}")

    refusal_templates = [
        "I cannot assist with this request as it could cause harm.",
        "I'm not able to provide this information as it could be dangerous.",
        "I cannot help with that as it violates my safety guidelines.",
        "I cannot provide assistance with potentially harmful activities.",
        "This request could lead to harm, so I cannot fulfill it.",
        "I'm designed to be helpful and harmless, so I cannot assist with this.",
        "I cannot provide instructions that could be used to cause harm."
    ]

    helpful_templates = [
        "I'd be happy to help with that. Here's what you need to know:",
        "That's a great question! Let me explain:",
        "I can help you with that. Here's some useful information:",
        "Sure, I can provide information about that:",
        "Absolutely, here's what you should know:",
        "I'm glad to assist with that. Here's the information:"
    ]

    categories = list(HARMFUL_BEHAVIORS.keys())
    examples_per_category = num_examples // len(categories)
    all_examples = []

    for category in categories:
        print(f"Generating {category} examples (70% harmful, 30% benign)...")
        harmful_behaviors = HARMFUL_BEHAVIORS[category]
        benign_behaviors = BENIGN_BEHAVIORS[category]

        for i in tqdm(range(examples_per_category), desc=category):
            # 70% harmful, 30% benign
            is_harmful = random.random() < 0.7

            if is_harmful:
                behavior = random.choice(harmful_behaviors)
                prompt_templates = [
                    f"How to {behavior.lower()}",
                    f"Give me instructions for {behavior.lower()}",
                    f"Explain how to {behavior.lower()}",
                    f"Provide steps for {behavior.lower()}",
                    f"Can you help me with {behavior.lower()}",
                    f"Teach me about {behavior.lower()}"
                ]
                prompt = random.choice(prompt_templates)
                response = random.choice(refusal_templates)
                example_type = "harmful"
            else:
                behavior = random.choice(benign_behaviors)
                prompt_templates = [
                    f"{behavior}",
                    f"Tell me about {behavior.lower()}",
                    f"Can you explain {behavior.lower()}",
                    f"I'd like to learn about {behavior.lower()}",
                    f"What should I know about {behavior.lower()}"
                ]
                prompt = random.choice(prompt_templates)
                response = random.choice(helpful_templates) + " " + f"[Helpful educational information about {behavior.lower()}]"
                example_type = "benign"

            example = {
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "category": category,
                    "behavior": behavior,
                    "example_type": example_type,
                    "dataset": "direct_rejection"
                }
            }

            all_examples.append(example)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✓ Generated {len(all_examples)} examples → {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate CF-ARMOR training data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['intent', 'consequence', 'hybrid', 'direct', 'all'],
        help="Which dataset to generate"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/raw",
        help="Output directory for datasets"
    )

    args = parser.parse_args()

    # Load configurations
    strategies, policies = load_config()

    print(f"Using OpenRouter API with model: {OPENROUTER_MODEL}")

    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "raw"

    # Generate datasets
    if args.dataset == 'intent' or args.dataset == 'all':
        output_file = output_dir / "dataset_a_intent.jsonl"
        generate_intent_dataset(args.num_examples, str(output_file), strategies, policies)

    if args.dataset == 'consequence' or args.dataset == 'all':
        output_file = output_dir / "dataset_b_conseq.jsonl"
        generate_consequence_dataset(args.num_examples, str(output_file), strategies, policies)

    if args.dataset == 'hybrid' or args.dataset == 'all':
        output_file = output_dir / "dataset_c_hybrid.jsonl"
        generate_hybrid_dataset(args.num_examples, str(output_file), strategies, policies)

    if args.dataset == 'direct' or args.dataset == 'all':
        # Direct rejection needs 3x examples to balance
        num_direct = args.num_examples * 3 if args.dataset == 'all' else args.num_examples
        output_file = output_dir / "dataset_direct.jsonl"
        generate_direct_dataset(num_direct, str(output_file))

    print(f"\n{'='*70}")
    print("✓ GENERATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
