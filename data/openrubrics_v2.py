"""
Preprocess the OpenRubrics/OpenRubric-v2 dataset to parquet format.
Filters high-quality conversations based on response length criteria.
"""

import argparse
import os
import random

import datasets
from transformers import AutoTokenizer
from tqdm import tqdm


def filter_sample(example, tokenizer, min_tokens=1024, max_tokens=4096):
    """
    Filter criteria for OpenRubrics samples:
    - Must have instruction, response_a, response_b, and winner
    - Both responses should be within token length range [min_tokens, max_tokens]
    - Winner must be either 'A' or 'B'
    """
    instruction = example.get("instruction", "")
    response_a = example.get("response_a", "")
    response_b = example.get("response_b", "")
    winner = example.get("winner", "")
    
    # Basic field validation
    if not instruction or not response_a or not response_b or not winner:
        return False, 0, 0
    
    # Winner must be 'A' or 'B'
    if winner not in ["A", "B"]:
        return False, 0, 0
    
    # Check token lengths for both responses
    tokens_a = len(tokenizer.encode(response_a, add_special_tokens=False))
    tokens_b = len(tokenizer.encode(response_b, add_special_tokens=False))
    
    # Both responses should be within the token range
    if not (min_tokens <= tokens_a <= max_tokens):
        return False, tokens_a, tokens_b
    
    if not (min_tokens <= tokens_b <= max_tokens):
        return False, tokens_a, tokens_b
    
    return True, tokens_a, tokens_b


def process_conversation(instruction, response_a, response_b):
    """
    Convert OpenRubrics format to standard prompt format.
    Returns prompt (instruction) and the two responses.
    """
    # Create prompt from instruction
    prompt = [{"role": "user", "content": instruction.strip()}]
    
    return prompt, response_a.strip(), response_b.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to extract (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--local_save_dir", 
        default="./data/openrubrics_v2", 
        help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B", help="Tokenizer to use for counting tokens")
    parser.add_argument(
        "--min_response_tokens", 
        type=int, 
        default=512, 
        help="Minimum response length in tokens"
    )
    parser.add_argument(
        "--max_response_tokens", 
        type=int, 
        default=4096, 
        help="Maximum response length in tokens"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Load tokenizer for token counting
    print(f"Loading tokenizer: {args.tokenizer}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    data_source = "OpenRubrics/OpenRubric-v2"
    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)
    
    # Load dataset - train split
    dataset = datasets.load_dataset(data_source, split="train")
    print(f"Total samples in dataset: {len(dataset)}", flush=True)

    # Filter and collect samples
    print(f"Filtering samples (both responses: {args.min_response_tokens} <= tokens <= {args.max_response_tokens})...", flush=True)
    
    processed_data = []
    preference_data = []
    seen_prompts = set()  # Avoid duplicates based on prompt content
    
    stats = {
        "total_processed": 0,
        "missing_fields": 0,
        "invalid_winner": 0,
        "duplicate_prompts": 0,
        "response_too_short": 0,
        "response_too_long": 0,
        "accepted": 0,
    }
    
    # Track token length distribution for analysis
    all_tokens_a = []
    all_tokens_b = []
    
    # Add progress bar
    pbar = tqdm(dataset, desc="Processing samples", total=len(dataset))
    
    for example in pbar:
        stats["total_processed"] += 1
        
        instruction = example.get("instruction", "")
        response_a = example.get("response_a", "")
        response_b = example.get("response_b", "")
        winner = example.get("winner", "")
        
        # Check for missing fields
        if not instruction or not response_a or not response_b or not winner:
            stats["missing_fields"] += 1
            continue
        
        # Check winner validity
        if winner not in ["A", "B"]:
            stats["invalid_winner"] += 1
            continue
        
        # Check for duplicate prompts
        prompt_hash = hash(instruction)
        if prompt_hash in seen_prompts:
            stats["duplicate_prompts"] += 1
            continue
        
        # Calculate token lengths for ALL samples (for statistics)
        tokens_a = len(tokenizer.encode(response_a, add_special_tokens=False))
        tokens_b = len(tokenizer.encode(response_b, add_special_tokens=False))
        all_tokens_a.append(tokens_a)
        all_tokens_b.append(tokens_b)
        
        # Filter by token length
        passed, _, _ = filter_sample(
            example, 
            tokenizer, 
            min_tokens=args.min_response_tokens,
            max_tokens=args.max_response_tokens
        )
        
        if not passed:
            # Categorize why it failed
            if tokens_a < args.min_response_tokens or tokens_b < args.min_response_tokens:
                stats["response_too_short"] += 1
            elif tokens_a > args.max_response_tokens or tokens_b > args.max_response_tokens:
                stats["response_too_long"] += 1
            continue
        
        # Update progress bar
        pbar.set_postfix({
            "accepted": stats["accepted"],
            "too_short": stats["response_too_short"],
            "too_long": stats["response_too_long"],
        })
        
        # All filters passed
        seen_prompts.add(prompt_hash)
        stats["accepted"] += 1
        
        # Process the conversation
        prompt, clean_response_a, clean_response_b = process_conversation(
            instruction, response_a, response_b
        )
        
        # Create training data entry
        data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "general",
            # For structure-only training, ground_truth is empty (no accuracy scoring)
            "reward_model": {"style": "structure", "ground_truth": ""},
            "extra_info": {
                "split": "train",
                "index": len(processed_data),
                "prompt_hash": prompt_hash,
            },
        }
        processed_data.append(data)
        
        # Create preference data entry
        # Determine chosen and rejected based on winner
        if winner == "A":
            chosen_response = clean_response_a
            rejected_response = clean_response_b
        else:  # winner == "B"
            chosen_response = clean_response_b
            rejected_response = clean_response_a
        
        # Format as LLaMA-Factory sharegpt preference format:
        # prompt: list of conversation history messages
        # chosen/rejected: single assistant message object
        preference_entry = {
            "prompt": prompt,
            "chosen": {"role": "assistant", "content": chosen_response},
            "rejected": {"role": "assistant", "content": rejected_response},
        }
        preference_data.append(preference_entry)
        
        # Progress reporting
        if len(processed_data) % 1000 == 0 and len(processed_data) > 0:
            print(f"\n  Collected {len(processed_data)} samples...", flush=True)
        
        # Stop if we have enough samples
        if args.num_samples and len(processed_data) >= args.num_samples:
            break
    
    pbar.close()

    print(f"\nFiltering complete!", flush=True)
    print(f"Statistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Missing fields: {stats['missing_fields']}")
    print(f"  Invalid winner: {stats['invalid_winner']}")
    print(f"  Duplicate prompts: {stats['duplicate_prompts']}")
    print(f"  Response too short: {stats['response_too_short']}")
    print(f"  Response too long: {stats['response_too_long']}")
    print(f"  Accepted: {stats['accepted']}")
    
    # Show token length distribution for ALL samples (to help adjust thresholds)
    if all_tokens_a and all_tokens_b:
        import numpy as np
        
        tokens_a_arr = np.array(all_tokens_a)
        tokens_b_arr = np.array(all_tokens_b)
        
        print(f"\nToken length distribution (ALL samples, before filtering):")
        print(f"  Response A:")
        print(f"    Min: {np.min(tokens_a_arr)}")
        print(f"    Max: {np.max(tokens_a_arr)}")
        print(f"    Mean: {np.mean(tokens_a_arr):.1f}")
        print(f"    Median: {np.median(tokens_a_arr):.0f}")
        print(f"    25th percentile: {np.percentile(tokens_a_arr, 25):.0f}")
        print(f"    75th percentile: {np.percentile(tokens_a_arr, 75):.0f}")
        print(f"    95th percentile: {np.percentile(tokens_a_arr, 95):.0f}")
        
        print(f"  Response B:")
        print(f"    Min: {np.min(tokens_b_arr)}")
        print(f"    Max: {np.max(tokens_b_arr)}")
        print(f"    Mean: {np.mean(tokens_b_arr):.1f}")
        print(f"    Median: {np.median(tokens_b_arr):.0f}")
        print(f"    25th percentile: {np.percentile(tokens_b_arr, 25):.0f}")
        print(f"    75th percentile: {np.percentile(tokens_b_arr, 75):.0f}")
        print(f"    95th percentile: {np.percentile(tokens_b_arr, 95):.0f}")
        
        print(f"\nSuggested token ranges based on data distribution:")
        print(f"  Conservative (keep ~50%): {int(np.percentile(tokens_a_arr, 25))}-{int(np.percentile(tokens_b_arr, 75))} tokens")
        print(f"  Moderate (keep ~75%): {int(np.percentile(tokens_a_arr, 10))}-{int(np.percentile(tokens_b_arr, 90))} tokens")
        print(f"  Liberal (keep ~90%): {int(np.percentile(tokens_a_arr, 5))}-{int(np.percentile(tokens_b_arr, 95))} tokens")
    
    # Calculate accepted samples statistics
    if processed_data:
        # Note: We can't get winner info from processed_data since we removed it from extra_info
        # This is consistent with ultrafeedback_10k.py which also doesn't store winner info
        print(f"\nAccepted samples: {len(processed_data)}")
        print(f"  (Winner distribution is stored in preference_data only)")


    # Shuffle
    indices = list(range(len(processed_data)))
    random.shuffle(indices)
    processed_data = [processed_data[i] for i in indices]
    preference_data = [preference_data[i] for i in indices]
    
    # Update indices after shuffle
    for idx, d in enumerate(processed_data):
        d["extra_info"]["index"] = idx

    # Split into train and eval
    EVAL_SIZE = min(100, len(processed_data) // 10)  # 10% or 100, whichever is smaller
    train_data = processed_data[EVAL_SIZE:]
    eval_data = processed_data[:EVAL_SIZE]
    train_preference_data = preference_data[EVAL_SIZE:]
    eval_preference_data = preference_data[:EVAL_SIZE]
    
    print(f"\nSplit data: {len(train_data)} train, {len(eval_data)} eval", flush=True)

    # Convert to HuggingFace Dataset
    train_dataset = datasets.Dataset.from_list(train_data)
    eval_dataset = datasets.Dataset.from_list(eval_data)

    # Save RL training data (parquet format)
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_output_path = os.path.join(local_dir, "train.parquet")
    eval_output_path = os.path.join(local_dir, "eval.parquet")
    train_dataset.to_parquet(train_output_path)
    eval_dataset.to_parquet(eval_output_path)
    print(f"\nSaved {len(train_dataset)} train samples to {train_output_path}", flush=True)
    print(f"Saved {len(eval_dataset)} eval samples to {eval_output_path}", flush=True)
    
    # Save preference data (chosen/rejected pairs) in separate folder
    import json
    preference_dir = os.path.join(local_dir, "preference_data")
    os.makedirs(preference_dir, exist_ok=True)
    
    train_pref_path = os.path.join(preference_dir, "train.jsonl")
    eval_pref_path = os.path.join(preference_dir, "eval.jsonl")
    
    with open(train_pref_path, "w", encoding="utf-8") as f:
        for entry in train_preference_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(train_preference_data)} train preference pairs to {train_pref_path}", flush=True)
    
    with open(eval_pref_path, "w", encoding="utf-8") as f:
        for entry in eval_preference_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(eval_preference_data)} eval preference pairs to {eval_pref_path}", flush=True)
    
    print(f"\n✓ Dataset preprocessing complete!", flush=True)

