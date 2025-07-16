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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/duplex_r1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--demo", action="store_true", help="Run typewriter demo only and exit.")

    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    data_source = "humanify/duplex_r1"
    dataset = datasets.load_dataset("open-r1/OpenR1-Math-220k", "default")

    full_dataset = dataset["train"]

    # Split into train and test (default 10% for test)
    split_ratio = 0.1  # 10% for test
    full_dataset = full_dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = full_dataset["train"]
    test_dataset = full_dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    if args.demo:
        # Typewriter demo: infinite loop, each time a new random problem, until Ctrl+C
        print("\n--- Typewriter Demo (Press Ctrl+C to exit) ---")
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
                    demo_rounds = simulate_typing(demo_problem, seed=None)
                    accumulated = ""
                    log_lines = []
                    for r in demo_rounds:
                        accumulated += r['input_text']
                        layout["text"].update(Panel(accumulated, title="User Input", border_style="bold green"))
                        log_line = (
                            f"[+{r['time_offset']:.1f}s] | type: {r['type']} | "
                            f"pause: {r['pause']:.2f}s | n_chars: {r['n_chars']} | "
                            f"max_output: {r['max_output']} | decode_rate: {r['decode_rate']:.2f} | text: {r['input_text']}"
                        )
                        log_lines.append(log_line)
                        log_panel = Panel("\n".join(log_lines[-10:]), title="Log", border_style="bold blue")
                        layout["log"].update(log_panel)
                        import time
                        time.sleep(r['pause'])
                    # After one demo, short pause and clear for next
                    import time
                    time.sleep(1.0)
                    accumulated = ""
                    log_lines = []
                    layout["text"].update(Panel("", title="User Input", border_style="bold green"))
                    layout["log"].update(Panel("", title="Log", border_style="bold blue"))
        except KeyboardInterrupt:
            print("\n--- End of Demo ---\n")
            exit(0)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")

            # Simulate typing rounds for duplex model
            typing_rounds = simulate_typing(problem + " " + instruction_following)

            answer = example.pop("answer")
            prompt = [
                {
                    "role": "system",
                    "content": 
"""You are a math expert. The user is progressively describing a problem. Based on the incomplete information available so far, you need to make temporary inferences. These temporary inferences will help you solve the problem more efficiently as more input arrives.

You are given a continuous stream of input about the problem, and you should always output your progressive, step-by-step inference. Your inference is temporary and may stop at any point, depending on the incoming input.

If you believe the problem is solved, output your final answer using the format: <answer>YOUR_MATH_ANSWER_WITHOUT_ANY_OTHER_TEXT</answer>.

If you have thought deeply but require more input to proceed, output <wait> at the end of your response."""
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

            # Get generations and correctness_math_verify from example
            generations = example.pop("generations", None)
            correctness_math_verify = example.pop("correctness_math_verify", None)

            # Select only verified generations
            verified_generations = []
            if generations is not None and correctness_math_verify is not None:
                verified_generations = [gen for gen, correct in zip(generations, correctness_math_verify) if correct]

            # Calculate token count for each verified generation
            generations_token_count_array = []
            for gen in verified_generations:
                token_count = len(tokenizer.encode(gen, add_special_tokens=False))
                generations_token_count_array.append(token_count)

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
                        "generations_max_token": max(generations_token_count_array) if generations_token_count_array else 4096,
                        "user_input_info": user_input_info_array
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
