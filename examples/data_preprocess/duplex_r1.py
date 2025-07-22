# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the duplex_r1 dataset to parquet format
"""

import argparse
import os
import re
import random
import time
import pickle
import hashlib

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Add tokenizer import
from transformers import AutoTokenizer

# For a better demo experience, use rich for terminal split view
try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("[INFO] rich not installed. Please install with: pip install rich")
    raise


def simulate_typing(
    problem,
    min_decode_rate=30,
    max_decode_rate=50,
    min_typing_pause=0.5,
    max_typing_pause=1,
    min_medium_pause=1.5,
    max_medium_pause=2.5,
    min_long_wait_pause=5,
    max_long_wait_pause=10,
    long_wait_prob=0.1,
    medium_wait_prob=0.3,
    paste_prob=0.02,
    paste_all_prob=0.01,
    min_paste_chars=10,
    max_paste_chars=30,
    min_typing_chars=2,
    max_typing_chars=8,
    seed=None
):
    """
    Simulate human typing and thinking pauses for a given problem string.
    Only char-level sampling is used. Each round samples a chunk of characters.
    Returns a list of dicts, each with: input_text, pause, max_output, time_offset, decode_rate, n_chars, type.
    """
    if seed is not None:
        random.seed(seed)
    full_str = problem
    rounds = []
    char_idx = 0
    time_offset = 0
    str_len = len(full_str)
    while char_idx < str_len:
        # Paste all behavior
        if random.random() < paste_all_prob:
            input_chunk = full_str[char_idx:]
            n_chars = len(input_chunk)
            char_idx = str_len
        # Paste chunk behavior
        elif random.random() < paste_prob:
            remaining_chars = str_len - char_idx
            if remaining_chars <= 0:
                break
            if remaining_chars < min_paste_chars:
                n_chars = remaining_chars
            else:
                n_chars = random.randint(min_paste_chars, min(max_paste_chars, remaining_chars))
            input_chunk = full_str[char_idx:char_idx+n_chars]
            n_chars = len(input_chunk)
            char_idx += n_chars
        # Normal typing (char-based)
        else:
            remaining_chars = str_len - char_idx
            if remaining_chars <= 0:
                break
            if remaining_chars < min_typing_chars:
                n_chars = remaining_chars
            else:
                n_chars = random.randint(min_typing_chars, min(max_typing_chars, remaining_chars))
            input_chunk = full_str[char_idx:char_idx+n_chars]
            n_chars = len(input_chunk)
            char_idx += n_chars
        # Decide pause type and duration
        r = random.random()
        if r < long_wait_prob:
            pause = random.uniform(min_long_wait_pause, max_long_wait_pause)
            pause_type = 'long_wait'
        elif r < long_wait_prob + medium_wait_prob:
            pause = random.uniform(min_medium_pause, max_medium_pause)
            pause_type = 'medium_wait'
        else:
            pause = random.uniform(min_typing_pause, max_typing_pause)
            pause_type = 'normal'
        decode_rate = random.uniform(min_decode_rate, max_decode_rate)
        max_output = int(pause * decode_rate)
        rounds.append({
            'input_text': input_chunk,
            'pause': pause,
            'max_output': max_output,
            'time_offset': time_offset,
            'decode_rate': decode_rate,
            'n_chars': n_chars,
            'type': pause_type
        })
        time_offset += pause
        if char_idx >= str_len:
            break
    return rounds


def simulate_typing_simple(
    problem,
    min_rounds=5,
    max_rounds=10,
    min_pause=0.5,
    max_pause=1.0,
    min_max_output=50,
    max_max_output=150,
    seed=None
):
    """
    Simulate simple typing by evenly splitting the problem into k rounds.
    Returns a list of dicts, each with: input_text, pause, max_output, time_offset, decode_rate, n_chars, type.
    """
    if seed is not None:
        random.seed(seed)
    
    full_str = problem
    str_len = len(full_str)
    
    # Randomly choose number of rounds between min_rounds and max_rounds
    k = random.randint(min_rounds, max_rounds)
    
    # Calculate characters per round (integer division)
    chars_per_round = str_len // k
    remaining_chars = str_len % k
    
    rounds = []
    char_idx = 0
    time_offset = 0
    
    for round_idx in range(k):
        # Distribute remaining characters to early rounds
        if round_idx < remaining_chars:
            n_chars = chars_per_round + 1
        else:
            n_chars = chars_per_round
            
        if n_chars <= 0:
            break
            
        input_chunk = full_str[char_idx:char_idx + n_chars]
        char_idx += n_chars
        
        # Consistent pause and max_output for all rounds
        pause = random.uniform(min_pause, max_pause)
        max_output = random.randint(min_max_output, max_max_output)
        decode_rate = max_output / pause  # Calculate decode rate based on pause and max_output
        
        rounds.append({
            'input_text': input_chunk,
            'pause': pause,
            'max_output': max_output,
            'time_offset': time_offset,
            'decode_rate': decode_rate,
            'n_chars': n_chars,
            'type': 'normal'
        })
        time_offset += pause
        
        if char_idx >= str_len:
            break
            
    return rounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/duplex_r1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--final_stage", action="store_true", default=False, help="Use format stage instead of final stage.")
    parser.add_argument("--max_ref_generation_token", default=2500)
    parser.add_argument("--min_problem_token", default=50)
    parser.add_argument("--max_problem_token", default=250)
    parser.add_argument("--max_answer_count", default=5)
    parser.add_argument("--answer_count_base_score", default=0.5)
    parser.add_argument("--ttfa_ratio_min", default=0.25)
    parser.add_argument("--ttfa_ratio_max", default=1)
    parser.add_argument("--ttfa_ratio_max_reward_point", default=0.8)
    parser.add_argument("--ttfa_ratio_base_score", default=0.5)
    parser.add_argument("--max_user_round", default=20)
    parser.add_argument("--cache_dir", default="~/.cache/duplex_r1", help="Directory to store token cache files")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of token cache")
    parser.add_argument("--demo", action="store_true", help="Run typewriter demo only and exit.")

    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    data_source = "humanify/duplex_r1"
    dataset = datasets.load_dataset("open-r1/OpenR1-Math-220k", "default")

    full_dataset = dataset["train"]
    print(f"Original dataset size: {len(full_dataset)}")

    def get_cache_key(tokenizer_path, dataset_size):
        """Generate a unique cache key based on tokenizer and dataset size"""
        key_string = f"{tokenizer_path}_{dataset_size}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cache_file_path(cache_key):
        """Get the cache file path for a given cache key"""
        cache_dir = os.path.expanduser(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"token_cache_{cache_key}.pkl")

    def load_token_cache(cache_key):
        """Load token cache from file if it exists"""
        cache_file = get_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            print(f"Loading token cache from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None

    def save_token_cache(cache_key, token_info_list):
        """Save token cache to file"""
        cache_file = get_cache_file_path(cache_key)
        print(f"Saving token cache to {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(token_info_list, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def get_verified_generations_token_info(generations, correctness_math_verify):
        """
        Get token information for verified generations.
        Returns (generations_token_list, avg_token_count)
        """
        generations_token = []
        if generations is not None and correctness_math_verify is not None:
            # Select only verified generations
            verified_generations = [gen for gen, correct in zip(generations, correctness_math_verify) if correct]
            
            # Calculate token count for each verified generation
            for gen in verified_generations:
                token_count = len(tokenizer.encode(gen, add_special_tokens=False))
                generations_token.append(token_count)
        
        avg_token_count = int(sum(generations_token) / len(generations_token)) if generations_token else 4096
        return generations_token, avg_token_count

    def precompute_token_info(example, idx):
        """
        Precompute token information for each example and cache it.
        Returns a dict with problem_tokens and avg_gen_tokens.
        """
        problem = example["problem"]
        problem_tokens = len(tokenizer.encode(problem, add_special_tokens=False))
        
        generations = example.get("generations", None)
        correctness_math_verify = example.get("correctness_math_verify", None)
        
        _, avg_gen_tokens = get_verified_generations_token_info(generations, correctness_math_verify)
        
        return {
            "problem_tokens": problem_tokens,
            "avg_gen_tokens": avg_gen_tokens,
            "index": idx
        }

    def filter_dataset_by_tokens_with_cache(example):
        """
        Filter dataset examples based on token constraints using cached token info.
        Returns True if the example should be kept, False otherwise.
        """
        # Use cached token information
        problem_tokens = example["problem_tokens"]
        avg_gen_tokens = example["avg_gen_tokens"]
        
        # Check problem token count range
        if problem_tokens < args.min_problem_token or problem_tokens > args.max_problem_token:
            return False
        
        if avg_gen_tokens > args.max_ref_generation_token:
            return False
        
        return True

    # Try to load token cache first
    cache_key = get_cache_key(args.tokenizer_path, len(full_dataset))
    cached_token_info = None if args.force_recompute else load_token_cache(cache_key)
    
    if cached_token_info is not None:
        print("Using cached token information")
        # Create a dataset with cached token info
        token_info_dataset = full_dataset.add_column("problem_tokens", [info["problem_tokens"] for info in cached_token_info])
        token_info_dataset = token_info_dataset.add_column("avg_gen_tokens", [info["avg_gen_tokens"] for info in cached_token_info])
        token_info_dataset = token_info_dataset.add_column("index", list(range(len(full_dataset))))
    else:
        # Precompute token information for all examples
        print("Precomputing token information for all examples...")
        token_info_dataset = full_dataset.map(precompute_token_info, with_indices=True, num_proc=16)
        
        # Save token cache for future use
        token_info_list = []
        for i in range(len(token_info_dataset)):
            token_info_list.append({
                "problem_tokens": token_info_dataset[i]["problem_tokens"],
                "avg_gen_tokens": token_info_dataset[i]["avg_gen_tokens"]
            })
        save_token_cache(cache_key, token_info_list)
    
    # Filter the dataset using cached token information
    print(f"Filtering dataset based on token constraints:")
    print(f"  - Problem tokens: [{args.min_problem_token}, {args.max_problem_token}]")
    print(f"  - Max generation tokens: <= {args.max_ref_generation_token}")
    filtered_dataset = token_info_dataset.filter(filter_dataset_by_tokens_with_cache, num_proc=16)
    
    # Get the filtered indices and add token info to the original dataset
    filtered_indices = filtered_dataset["index"]
    full_dataset = full_dataset.select(filtered_indices)
    
    # Add token information to the filtered dataset for later use
    def add_token_info_to_dataset(example, idx):
        # Get the corresponding token info from filtered dataset
        token_info = filtered_dataset[idx]
        example["problem_tokens"] = token_info["problem_tokens"]
        example["avg_gen_tokens"] = token_info["avg_gen_tokens"]
        return example
    
    full_dataset = full_dataset.map(add_token_info_to_dataset, with_indices=True, num_proc=16)
    
    print(f"Filtered dataset size: {len(full_dataset)}")
    print(f"Removed {len(dataset['train']) - len(full_dataset)} examples due to token constraints")

    # Split into train and test (default 10% for test)
    split_ratio = 0.1  # 10% for test
    full_dataset = full_dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = full_dataset["train"]
    test_dataset = full_dataset["test"]

    instruction_following = "Now think step by step and output the final answer."

    def run_typewriter_demo(use_simple=False):
        """Run typewriter demo with either complex or simple typing simulation"""
        demo_type = "Simple" if use_simple else "Complex"
        print(f"\n--- {demo_type} Typewriter Demo (Press Ctrl+C to exit) ---")
        console = Console()
        layout = Layout()
        layout.split_column(
            Layout(name="text", ratio=2),
            Layout(name="log", ratio=1)
        )
        try:
            with Live(layout, refresh_per_second=10, console=console, screen=True):
                while True:
                    random_idx = random.randint(0, len(train_dataset)-1)
                    demo_problem = train_dataset[random_idx]["problem"]
                    
                    # Choose typing simulation function
                    if use_simple:
                        demo_rounds = simulate_typing_simple(demo_problem, seed=None)
                    else:
                        demo_rounds = simulate_typing(demo_problem, seed=None)
                    
                    accumulated = ""
                    log_lines = []
                    for r in demo_rounds:
                        accumulated += r['input_text']
                        layout["text"].update(Panel(accumulated, title=f"User Input ({demo_type})", border_style="bold green"))
                        log_line = (
                            f"[+{r['time_offset']:.1f}s] | type: {r['type']} | "
                            f"pause: {r['pause']:.2f}s | n_chars: {r['n_chars']} | "
                            f"max_output: {r['max_output']} | decode_rate: {r['decode_rate']:.2f} | text: {r['input_text']}"
                        )
                        log_lines.append(log_line)
                        log_panel = Panel("\n".join(log_lines[-10:]), title=f"Log ({demo_type})", border_style="bold blue")
                        layout["log"].update(log_panel)
                        import time
                        time.sleep(r['pause'])
                    # After one demo, short pause and clear for next
                    import time
                    time.sleep(1.0)
                    accumulated = ""
                    log_lines = []
                    layout["text"].update(Panel("", title=f"User Input ({demo_type})", border_style="bold green"))
                    layout["log"].update(Panel("", title=f"Log ({demo_type})", border_style="bold blue"))
        except KeyboardInterrupt:
            print(f"\n--- End of {demo_type} Demo ---\n")
            exit(0)

    if args.demo:
        run_typewriter_demo(use_simple=not args.final_stage)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")

            problem_with_instruction = problem + "\n\n" + instruction_following
            typing_rounds = simulate_typing(problem_with_instruction) if args.final_stage else simulate_typing_simple(problem_with_instruction)

            answer = example.pop("answer")
            prompt = [
                {
                    "role": "system",
                    "content": 
f"""You are a math expert skilled in step-by-step reasoning.

Your user is a streaming input system that continuously provides incomplete information about a math problem — sometimes the input may be empty. Based on the input received so far, you should perform interim reasoning steps. These steps will help you solve the problem more efficiently as additional input arrives.

In each reply, continue your reasoning from where you left off.

Your reasoning is provisional and may be interrupted at any time, depending on the input stream.

If you believe the problem is solved, output your considered answer using the format: <answer>YOUR_MATH_ANSWER_WITHOUT_ANY_OTHER_TEXT</answer>. You have {args.max_answer_count} attempts to output and revise your considered answer.

If you've thought through everything possible with the current input and need more information, you can end your response and wait for the next input. If you have nothing to think or output, you can reply with an empty message."""
                },
            ]
            user_input_info_array = []
            for i, r in enumerate(typing_rounds):
                if i == 0:
                    prompt.append({
                        "role": "user",
                        "content": r["input_text"]
                    })
                
                user_input_info_array.append(r)

            # Use cached token information
            avg_gen_tokens = example.pop("avg_gen_tokens", 4096)
            # Remove cached problem_tokens as it's not needed in final output
            example.pop("problem_tokens", None)

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "interaction_kwargs": {
                        "name": "duplex_r1",
                        "problem": problem,
                        "ground_truth": answer,
                        "avg_gen_tokens": avg_gen_tokens,
                        "user_input_info": user_input_info_array,
                        "max_answer_count": args.max_answer_count,
                        "answer_count_base_score": args.answer_count_base_score,
                        "ttfa_ratio_min": args.ttfa_ratio_min,
                        "ttfa_ratio_max": args.ttfa_ratio_max,
                        "ttfa_ratio_max_reward_point": args.ttfa_ratio_max_reward_point,
                        "ttfa_ratio_base_score": args.ttfa_ratio_base_score,
                        "stage": "final" if args.final_stage else "format",
                        "max_user_round": args.max_user_round
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=16, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=16, remove_columns=test_dataset.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
