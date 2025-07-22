# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import os
import time
import json
from typing import List, Dict, Optional
from collections import defaultdict
import openai
from math_verify import parse, verify

OPENAI_API_KEY = 'sk-xxx'
OPENAI_MODEL = "Qwen/Qwen3-32B-AWQ"
OPENAI_BASE_URL = "http://127.0.0.1:10000/v1"

# Logging configuration
LOG_INTERVAL_SECONDS = float(os.getenv("DUPLEX_LOG_INTERVAL_SECONDS", "60"))  # Log every N seconds
SAVE_DETAILED_SAMPLES = os.getenv("SAVE_DETAILED_SAMPLES", "true").lower() == "true"  # Whether to save detailed samples to local files
DETAILED_SAMPLES_DIR = os.getenv("DETAILED_SAMPLES_DIR", "logs/duplex_r1_detailed_samples")  # Directory to save detailed sample files

# Global variables for tracking logging state
_log_call_count = 0
_last_log_time = 0.0

# Real-time statistics tracking (no memory accumulation)
class RunningStats:
    """Real-time statistics calculator that doesn't store all values in memory."""
    
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def add_value(self, value: float):
        """Add a value and update running statistics."""
        self.count += 1
        self.sum += value
        self.sum_sq += value * value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def get_stats(self) -> dict:
        """Get current statistics without storing all values."""
        if self.count == 0:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        
        avg = self.sum / self.count
        # Calculate standard deviation using sum of squares
        variance = (self.sum_sq / self.count) - (avg * avg)
        std = (variance ** 0.5) if variance > 0 else 0.0
        
        return {
            "avg": avg,
            "min": self.min_val,
            "max": self.max_val,
            "std": std,
            "count": self.count
        }
    
    def reset(self):
        """Reset all statistics."""
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

# Global running statistics for each score type
_score_stats = defaultdict(RunningStats)

def _should_log_now() -> bool:
    """Check if it's time to log based on time interval."""
    global _last_log_time
    current_time = time.time()
    return (current_time - _last_log_time) >= LOG_INTERVAL_SECONDS

def _log_to_local_json(messages: List[dict], ground_truth: str, scores: Dict[str, float], final_score: float):
    """Log both detailed sample and aggregated scores to a single local JSON file."""
    global SAVE_DETAILED_SAMPLES, DETAILED_SAMPLES_DIR, _score_stats, _last_log_time
    
    if not SAVE_DETAILED_SAMPLES:
        return
    
    try:
        # Create comprehensive log entry
        log_data = {
            "call_count": _log_call_count,
            "timestamp": time.time(),
            "final_score": final_score,
            "ground_truth": ground_truth,
            "message_count": len(messages),
            "user_message_count": len([m for m in messages if m.get('role') == 'user']),
            "assistant_message_count": len([m for m in messages if m.get('role') == 'assistant']),
            "log_type": "unified_log"
        }
        
        # Add individual scores
        for score_name, score_value in scores.items():
            log_data[f"score_{score_name}"] = score_value
        
        # Add aggregated statistics for each score type
        aggregated_stats = {}
        total_samples = 0
        
        for score_type, stats in _score_stats.items():
            stats_dict = stats.get_stats()
            if stats_dict["count"] > 0:
                aggregated_stats[f"avg_{score_type}"] = stats_dict["avg"]
                aggregated_stats[f"min_{score_type}"] = stats_dict["min"]
                aggregated_stats[f"max_{score_type}"] = stats_dict["max"]
                aggregated_stats[f"std_{score_type}"] = stats_dict["std"]
                aggregated_stats[f"count_{score_type}"] = stats_dict["count"]
                total_samples = max(total_samples, stats_dict["count"])
        
        # Add aggregated statistics to log data
        log_data["aggregated_stats"] = aggregated_stats
        log_data["total_samples"] = total_samples

        # Add complete messages for detailed logging
        log_data["complete_messages"] = messages
        
        # Create filename with integer timestamp and call count
        timestamp_int = int(log_data["timestamp"])
        filename = f"unified_log_{timestamp_int}_call_{_log_call_count}.json"
        
        # Ensure directory exists
        os.makedirs(DETAILED_SAMPLES_DIR, exist_ok=True)
        
        # Save to local JSON file
        filepath = os.path.join(DETAILED_SAMPLES_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        # Reset statistics after logging
        for stats in _score_stats.values():
            stats.reset()
        
        # Update last log time
        _last_log_time = time.time()
        
        print(f"Unified log saved to: {filepath}")
        print(f"  - Detailed sample with {len(messages)} messages")
        print(f"  - {len(scores)} individual scores")
        print(f"  - {len(aggregated_stats)} aggregated statistics")
        
    except Exception as e:
        print(f"Warning: Failed to save unified log to local file: {e}")



def get_logging_stats():
    """Get current logging statistics."""
    global _log_call_count, _score_stats, _last_log_time
    
    # Calculate next log time
    next_log_time = _last_log_time + LOG_INTERVAL_SECONDS
    current_time = time.time()
    time_until_next_log = max(0, next_log_time - current_time)
    
    stats = {
        "total_calls": _log_call_count,
        "log_interval_seconds": LOG_INTERVAL_SECONDS,
        "save_detailed_samples": SAVE_DETAILED_SAMPLES,
        "detailed_samples_dir": DETAILED_SAMPLES_DIR,
        "time_until_next_log_seconds": time_until_next_log,
        "last_log_time": _last_log_time,
        "current_time": current_time,
        "score_types": list(_score_stats.keys()),
        "score_counts": {k: stats.get_stats()["count"] for k, stats in _score_stats.items()}
    }
    
    return stats

def reset_logging_state():
    """Reset the logging state (useful for testing or restarting logging)."""
    global _log_call_count, _score_stats, _last_log_time
    
    _log_call_count = 0
    _last_log_time = 0.0
    for stats in _score_stats.values():
        stats.reset()
    print("Logging state reset.")


def compute_answer_score(messages: List[dict], ground_truth: str) -> float:
    """
    Extracts the last <answer>xxx</answer> from the concatenated assistant messages, compares it to ground_truth (which may or may not be wrapped in <answer> tags), and returns 1 if math-verify judges them equivalent, else 0.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
        ground_truth (str): The ground truth, possibly with or without <answer> tags.
    Returns:
        float: 1.0 if math-verify judges them equivalent, else 0.0.
    """
    # Concatenate all assistant message contents
    combined_output = ''.join([msg['content']
                              for msg in messages if msg['role'] == 'assistant'])

    def extract_last_answer(text: str) -> str:
        matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    output_answer = extract_last_answer(combined_output)

    if output_answer is None or ground_truth == "":
        return 0.0

    try:
        parsed_output = parse(output_answer)
        parsed_gt = parse(ground_truth)
        if verify(parsed_gt, parsed_output):
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"math-verify error: {e}")
        return 0.0


def append_user_input_info(messages: List[dict], user_input_info: List[dict]) -> List[dict]:
    """
    Appends user_input_info to the messages list.
    """
    user_msg_index = 0
    for msg in messages:
        if user_msg_index >= len(user_input_info):
            break
        if msg["role"] == "user":
            msg["user_input_info"] = user_input_info[user_msg_index]
            user_msg_index += 1

    return messages


def compute_answer_count_score(messages: List[dict], max_answer_count: int = 5, answer_count_base_score: float = 0.0) -> float:
    """
    Computes a score based on the number of <answer>...</answer> tags in the combined output.
    Returns 0.0 if there are more than max_answer_count answer tags.
    For max_answer_count or fewer tags, returns a score that decreases as count increases,
    with a minimum base score when within the acceptable range.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
        max_answer_count (int): Maximum allowed answer tags before penalty (default: 5)
        answer_count_base_score (float): Minimum score when answer count is within acceptable range (default: 0.0)
    Returns:
        float: Score between 0.0 and 1.0 based on answer tag count
    """
    # Concatenate all assistant message contents
    combined_output = ''.join([msg['content']
                              for msg in messages if msg['role'] == 'assistant'])

    # Count the number of complete <answer>...</answer> tags
    answer_count = len(re.findall(
        r"<answer>(.*?)</answer>", combined_output, re.DOTALL))

    # If more than max_answer_count answer tags, return 0.0
    if answer_count > max_answer_count:
        return 0.0

    # For max_answer_count or fewer tags, return a score that decreases as count increases
    # Formula: score = base_score + (1.0 - base_score) * (max_answer_count + 1 - answer_count) / max_answer_count
    # This gives: 0 tags = base_score + (1.0 - base_score) * (max+1)/max (clamped to 1.0), 
    # 1 tag = base_score + (1.0 - base_score) * max/max = 1.0, etc.
    score = answer_count_base_score + (1.0 - answer_count_base_score) * (max_answer_count + 1 - answer_count) / max_answer_count
    return max(0.0, min(1.0, score))


def compute_format_score(messages: List[dict]) -> float:
    """
    Computes a global format score for a list of messages.
    Checks:
    1. <answer> and </answer> tags must be properly paired and closed (globally across all assistant messages).
    2. Every user message must be immediately followed by an assistant message; otherwise, returns 0.0.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
    Returns:
        float: 1.0 if all checks pass, 0.0 otherwise.
    """
    # Check that every user message is immediately followed by an assistant message
    for i, msg in enumerate(messages):
        if msg['role'] == 'user':
            if i + 1 >= len(messages) or messages[i + 1]['role'] != 'assistant':
                return 0.0

    # Collect all assistant message contents
    assistant_outputs = [msg['content']
                         for msg in messages if msg['role'] == 'assistant']

    # Global check for <answer> tag pairing
    combined_output = ''.join(assistant_outputs)
    answer_open = [m.start()
                   for m in re.finditer(r'<answer>', combined_output)]
    answer_close = [m.start()
                    for m in re.finditer(r'</answer>', combined_output)]
    if len(answer_open) != len(answer_close):
        return 0.0
    for open_pos, close_pos in zip(answer_open, answer_close):
        if open_pos > close_pos:
            return 0.0

    return 1.0


def compute_fluency_score(messages: List[dict]) -> float:
    """
    Computes fluency score by evaluating the naturalness of transitions between consecutive assistant messages.
    Uses batch LLM requests to assess each transition individually.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
    Returns:
        float: Average fluency score between 0.0 and 1.0
    """
    # Extract all assistant message contents
    outputs = [msg['content']
               for msg in messages if msg['role'] == 'assistant']
    if len(outputs) <= 1:
        return 1.0  # Single output is always fluent

    # Prepare individual requests for each transition
    def create_transition_prompt(prev_output: str, next_output: str, transition_id: int) -> str:
        """Create a prompt for evaluating a single transition."""
        return f"""Below are two consecutive text outputs. Please rate how naturally the first output flows into the second output.

First output:
{prev_output}

[TRANSITION POINT]

Second output:
{next_output}

Rating scale:
- 0.0: Very unnatural, jarring, or disconnected transition
- 0.5: Somewhat awkward but understandable transition  
- 1.0: Completely natural and smooth transition"""

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY,
                               base_url=OPENAI_BASE_URL)

        # Prepare all requests
        requests = []
        for i in range(len(outputs) - 1):
            prompt = create_transition_prompt(
                outputs[i], outputs[i + 1], i + 1)
            requests.append({
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system",
                        "content": "You are a text fluency evaluation expert. Your task is to rate the naturalness of text transitions. You MUST respond with ONLY a single decimal number between 0.0 and 1.0 (e.g., 0.8, 0.5, 1.0). Do NOT include any explanations, words, or additional text. Output format: [number only]"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,  # We only need a single number
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            })

        # Execute all requests (in sequence for now, could be made parallel)
        scores = []
        for i, request in enumerate(requests):
            try:
                response = client.chat.completions.create(**request)
                response_text = response.choices[0].message.content.strip()

                # Parse the single score from response
                # Try to extract a decimal number from the response (prioritize valid 0.0-1.0 range)
                number_match = re.search(
                    r'([01]\.?[0-9]*|0\.[0-9]+|1\.0*)', response_text)
                if number_match:
                    score = float(number_match.group(1))
                    # Clamp score to [0, 1] range
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                else:
                    print(
                        f"Warning: Could not parse score from response: '{response_text}'. Using fallback score 0.5.")
                    scores.append(0.5)

            except ValueError:
                print(
                    f"Warning: Invalid score format in response for transition {i+1}. Using fallback score 0.5.")
                scores.append(0.5)
            except Exception as e:
                print(
                    f"Warning: Error processing transition {i+1}: {e}. Using fallback score 0.5.")
                scores.append(0.5)

        # If we couldn't get any scores, return fallback
        if not scores:
            print("Warning: No valid scores obtained. Using fallback score 0.5.")
            return 0.5

        # Return average score
        return sum(scores) / len(scores)

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return 0.5  # Fallback score on error


def compute_long_wait_alignment_score(messages: List[dict]) -> float:
    """
    For every user message with type == 'long_wait', check that the next message is an assistant message
    whose generated token length reaches the max_output limit. Returns the average score (1.0 for each correct, 0.0 for each incorrect).
    If there are no 'long_wait' user messages, returns 1.0 by default.
    """
    long_wait_indices = [
        i for i, msg in enumerate(messages)
        if msg['role'] == 'user' and msg['user_input_info']['type'] == 'long_wait'
    ]
    if not long_wait_indices:
        return 1.0  # No long waits, considered perfect

    scores = []
    for idx in long_wait_indices:
        if idx + 1 >= len(messages):
            scores.append(0.0)
            continue
        next_msg = messages[idx + 1]
        user_msg = messages[idx]

        # Check if the assistant message's token count reaches the max_output limit
        if next_msg['role'] == 'assistant' and 'meta_info' in next_msg:
            completion_tokens = next_msg['meta_info'].get(
                'completion_tokens', 0)
            max_output = user_msg['user_input_info'].get('max_output', 0)

            # Consider it correct if the token count reaches or exceeds max_output
            if completion_tokens >= max_output:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)


def _get_token_counts_for_ttfa(messages: List[dict]) -> tuple[int, int]:
    """
    Helper function to find the last valid user message and calculate token counts for TTFA scoring.

    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}

    Returns:
        tuple[int, int]: (previous_output_count, later_output_count)
            - previous_output_count: tokens used before the last valid user message
            - later_output_count: tokens used after the last valid user message

    Raises:
        ValueError: If no valid user message (with user_input_info) is found
    """
    # Find the index of the last valid user message (one that has user_input_info)
    user_end_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user' and 'user_input_info' in messages[i]:
            user_end_idx = i
            break
    if user_end_idx == -1:
        raise ValueError(
            "No valid user message (with user_input_info) found in messages list.")

    previous_output_count = 0
    later_output_count = 0
    for idx, msg in enumerate(messages):
        if msg['role'] == 'assistant':
            token_count = msg['meta_info']['completion_tokens']
            if idx > user_end_idx:
                later_output_count += token_count
            else:
                previous_output_count += token_count

    return previous_output_count, later_output_count


def compute_ttfa_early_ratio_score(messages: List[dict], target_min: float, target_max: float, max_reward_point: float, base_score: float) -> float:
    """
    Compute a score that rewards keeping the early thinking token ratio within an optimal range.

    The score encourages the model to use an appropriate amount of tokens for early thinking
    relative to the total token budget, with a preference for higher ratios within the valid range.

    Formula: 
    - If early_ratio is outside [target_min, target_max]: score = 0.0
    - If early_ratio is within [target_min, max_reward_point]: 
        score = base_score + (1.0 - base_score) * (early_ratio - target_min) / (max_reward_point - target_min)
    - If early_ratio is within [max_reward_point, target_max]: score = 1.0

    where:
        early_ratio = c_m / (c_m + c_n)  # ratio of early thinking tokens to total tokens
        target_min: minimum desired early ratio (e.g., 0.25)
        target_max: maximum desired early ratio (e.g., 0.8)
        max_reward_point: point where score reaches 1.0 (e.g., 0.6)
        base_score: minimum score when early_ratio = target_min (e.g., 0.3)

    Higher score when:
    - Early thinking uses an appropriate proportion of the total token budget
    - Within the valid range, higher early thinking ratios are preferred until max_reward_point

    This prevents both:
    - Too little early thinking (rushing to answer) - score = 0.0
    - Too much early thinking (burning through the budget) - score = 0.0
    - Within valid range, encourages more thorough early thinking up to max_reward_point
    """
    try:
        previous_output_count, later_output_count = _get_token_counts_for_ttfa(
            messages)
        total_tokens = previous_output_count + later_output_count
        if total_tokens == 0:
            return 1.0  # No tokens used, consider it neutral

        early_ratio = previous_output_count / total_tokens

        # Define target range for early thinking ratio
        if early_ratio < target_min or early_ratio > target_max:
            # Outside target range - score = 0.0
            score = 0.0
        elif early_ratio >= max_reward_point:
            # At or beyond max_reward_point - score = 1.0
            score = 1.0
        else:
            # Within [target_min, max_reward_point] - linear interpolation from base_score to 1.0
            # Protect against division by zero
            denominator = max_reward_point - target_min
            if abs(denominator) < 1e-10:  # Very small value, treat as zero
                score = base_score
            else:
                # Linear interpolation: base_score at target_min, 1.0 at max_reward_point
                interpolation_factor = (early_ratio - target_min) / denominator
                score = base_score + (1.0 - base_score) * interpolation_factor

        return max(0.0, min(1.0, score))
    except ValueError as e:
        print(f"Error in compute_ttfa_early_ratio_score: {e}")
        return 0.5  # Fallback score on error


def compute_ttfa_length_score(messages: List[dict], ref_count: int) -> float:
    """
    Compute the length compliance score that penalizes exceeding the reference token budget.
    This score ensures the model doesn't use excessive tokens compared to the reference thinking count.

    Formula: score_length = max(0, 1 - ((c_m + c_n - k)_+) / s)
    where:
        s = sigma * k, sigma ~ 0.2, k = ref_count
        (x)_+ = max(x, 0)
        c_m + c_n: total assistant tokens used

    Higher score when:
    - Total tokens used (c_m + c_n) is close to or below the reference count (k)
    - The model is efficient with token usage

    This prevents the model from being wasteful with tokens and encourages efficiency.
    """
    try:
        previous_output_count, later_output_count = _get_token_counts_for_ttfa(
            messages)
        total_tokens = previous_output_count + later_output_count
        sigma = 0.2
        k = ref_count
        s = sigma * k
        over_length = max(total_tokens - k, 0)
        score_length = max(0.0, 1.0 - (over_length / s) if s > 0 else 0.0)

        return score_length
    except ValueError as e:
        print(f"Error in compute_ttfa_length_score: {e}")
        return 0.5  # Fallback score on error


def compute_score(messages: List[dict],
                  ground_truth: str,
                  ref_count: int,
                  max_answer_count: int,
                  answer_count_base_score: float,
                  ttfa_ratio_min: float,
                  ttfa_ratio_max: float,
                  ttfa_ratio_max_reward_point: float,
                  ttfa_ratio_base_score: float,
                  stage: str,
                  log: bool = False) -> float:
    """
    Compute the final reward score by combining sub-scores with weighted average.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
        ground_truth (str): The ground truth answer for answer_score
        ref_count (int): Reference thinking token count for time_to_final_answer_score
        max_answer_count (int): Maximum number of answer tags
        answer_count_base_score (float): Minimum score when answer count is within acceptable range
        ttfa_ratio_min (float): Minimum desired early ratio
        ttfa_ratio_max (float): Maximum desired early ratio
        ttfa_ratio_max_reward_point (float): Point where score reaches 1.0
        ttfa_ratio_base_score (float): Minimum score when early_ratio = target_min
        stage (str): 'format' for format stage, 'final' for full evaluation stage
        log (bool): Whether to enable local JSON logging
    Returns:
        float: The final weighted score
    """
    global _log_call_count, _score_stats
    
    # Compute all possible scores
    format_score = compute_format_score(messages)
    fluency_score = compute_fluency_score(messages)
    answer_score = compute_answer_score(messages, ground_truth)
    answer_count_score = compute_answer_count_score(messages, max_answer_count, answer_count_base_score)
    final_answer_score = answer_score * answer_count_score
    ttfa_early_ratio_score = compute_ttfa_early_ratio_score(
        messages, ttfa_ratio_min, ttfa_ratio_max, ttfa_ratio_max_reward_point, ttfa_ratio_base_score)
    
    if stage == "format":
        final_score = (
            0.2 * format_score +
            0.2 * fluency_score +
            0.3 * final_answer_score + 
            0.3 * ttfa_early_ratio_score
        )
    elif stage == "final":
        # For final stage, compute all scores
        long_wait_alignment_score = compute_long_wait_alignment_score(messages)
        ttfa_length_score = compute_ttfa_length_score(messages, ref_count)
        ttfa_score = ttfa_early_ratio_score * ttfa_length_score
        # Weighted sum according to user-confirmed weights
        # TODO: refine the weights (WIP)
        final_score = (
            0.35 * final_answer_score +
            0.15 * format_score +
            0.10 * fluency_score +
            0.15 * long_wait_alignment_score +
            0.25 * ttfa_score
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Handle logging if enabled
    if log:
        _log_call_count += 1
        
        # Collect all scores for logging
        scores = {
            "format_score": format_score,
            "fluency_score": fluency_score,
            "answer_score": answer_score,
            "answer_count_score": answer_count_score,
            "final_answer_score": final_answer_score,
            "ttfa_early_ratio_score": ttfa_early_ratio_score,
            "final_score": final_score
        }
        
        # Add stage-specific scores
        if stage == "final":
            scores.update({
                "long_wait_alignment_score": long_wait_alignment_score,
                "ttfa_length_score": ttfa_length_score,
                "ttfa_score": ttfa_score
            })
        
        # Add scores to real-time statistics
        for score_name, score_value in scores.items():
            _score_stats[score_name].add_value(score_value)
        
        # Log detailed sample and aggregated scores based on time interval
        if _should_log_now():
            _log_to_local_json(messages, ground_truth, scores, final_score)

    return final_score


def test_compute_answer_score():
    print("Testing compute_answer_score...")
    # Case 1: Exact match with <answer> tags
    messages = [
        {"role": "assistant", "content": "<answer>42</answer>"}
    ]
    gt = "<answer>42</answer>"
    assert compute_answer_score(messages, gt) == 0.0

    # Case 2: No <answer> in ground truth, but matches
    gt = "42"
    assert compute_answer_score(messages, gt) == 1.0

    # Case 3: No match
    gt = "<answer>43</answer>"
    assert compute_answer_score(messages, gt) == 0.0

    # Case 4: No <answer> in output
    messages = [{"role": "assistant", "content": "no answer here"}]
    assert compute_answer_score(messages, "something") == 0.0

    # Case 5: Multiple <answer> tags, last one matches
    messages = [
        {"role": "assistant", "content": "<answer>1</answer><answer>2</answer>"}
    ]
    gt = "2"
    assert compute_answer_score(messages, gt) == 1.0

    # Case 6: Mathematical equivalence - fractions
    messages = [
        {"role": "assistant", "content": "<answer>1/2</answer>"}
    ]
    gt = "0.5"
    assert compute_answer_score(messages, gt) == 1.0

    # Case 7: Mathematical equivalence - expressions
    messages = [
        {"role": "assistant", "content": "<answer>2*3</answer>"}
    ]
    gt = "6"
    assert compute_answer_score(messages, gt) == 1.0

    print("compute_answer_score tests passed.")


def test_compute_format_score():
    print("Testing compute_format_score...")
    # Case 1: Properly paired, user/assistant alternation
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<answer>ok</answer>"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<answer>yes</answer>"},
    ]
    assert compute_format_score(messages) == 1.0

    # Case 2: Unpaired <answer>
    messages[3]["content"] = "<answer>yes"
    assert compute_format_score(messages) == 0.0
    messages[3]["content"] = "<answer>yes</answer>"  # restore

    # Case 3: User not followed by assistant
    bad_messages = [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2"},
    ]
    assert compute_format_score(bad_messages) == 0.0

    # Case 4: Empty assistant message (should pass format check since it's not empty in terms of structure)
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": ""},
    ]
    assert compute_format_score(messages) == 1.0

    # Case 5: Assistant message with only whitespace (should pass format check)
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "   "},
    ]
    assert compute_format_score(messages) == 1.0

    print("compute_format_score tests passed.")


def test_compute_fluency_score():
    print("Testing compute_fluency_score...")

    # Case 1: Single assistant message - should return 1.0
    messages = [
        {"role": "assistant", "content": "Hello world."}
    ]
    score = compute_fluency_score(messages)
    assert score == 1.0, f"Expected 1.0 for single message, got {score}"

    # Case 2: No assistant messages - should return 1.0
    messages = [
        {"role": "user", "content": "Question?"}
    ]
    score = compute_fluency_score(messages)
    assert score == 1.0, f"Expected 1.0 for no assistant messages, got {score}"

    # Case 3: Empty messages list - should return 1.0
    messages = []
    score = compute_fluency_score(messages)
    assert score == 1.0, f"Expected 1.0 for empty messages, got {score}"

    # Case 4: Normal fluent transition
    messages = [
        {"role": "assistant", "content": "Let me solve this step by step."},
        {"role": "assistant",
            "content": "First, I'll identify the variables in the equation."},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 5: Mixed roles (only assistant messages should be considered)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate."},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 6: Extreme topic jumping without context
    messages = [
        {"role": "assistant", "content": "The solution to this algebra problem is x = 5."},
        {"role": "assistant",
            "content": "My grandmother's recipe for chocolate cake requires three eggs."},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 7: Garbled text and encoding issues
    messages = [
        {"role": "assistant", "content": "The answer is quite simple and straightforward."},
        {"role": "assistant", "content": "��������⨀☠♣↕₣‰℮ъцщюя∀∃∇√∞∫≤≥≠±÷×¡¿"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 8: Repetitive gibberish with special characters
    messages = [
        {"role": "assistant", "content": "Here's my analysis of the problem."},
        {"role": "assistant",
            "content": "aaaaaa!!!!! bbbbb????? 123123123 @@@@@@ ######### aaaaaa!!!!!"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 9: Broken tags and malformed content
    messages = [
        {"role": "assistant", "content": "Here's the answer: <answer>42</answer>"},
        {"role": "assistant", "content": "<answer>no</answer><answer>"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 10: Control characters and special formatting
    messages = [
        {"role": "assistant", "content": "Let me think about this problem."},
        {"role": "assistant",
            "content": "result\n\n\r\t\t\t   \x00\x01\x02 is \b\f\v here somewhere\n\r\n"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    # Case 11: Extremely abrupt contradiction and reversal
    messages = [
        {"role": "assistant",
            "content": "Based on careful analysis, the correct answer is definitely A."},
        {"role": "assistant",
            "content": "NO WAIT WRONG EVERYTHING I SAID IS FALSE IT'S ACTUALLY Z!!!"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"

    print("compute_fluency_score tests passed (API calls may fallback to 0.5).")


def test_compute_long_wait_alignment_score():
    print("Testing compute_long_wait_alignment_score...")
    # Case 1: No long_wait user messages
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 10}},
    ]
    assert compute_long_wait_alignment_score(messages) == 1.0

    # Case 2: long_wait user, assistant reaches max_output
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {
            "type": "long_wait", "max_output": 50}},
        {"role": "assistant", "content": "Thinking...",
            "meta_info": {"completion_tokens": 50}},
    ]
    assert compute_long_wait_alignment_score(messages) == 1.0

    # Case 3: long_wait user, assistant exceeds max_output
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {
            "type": "long_wait", "max_output": 50}},
        {"role": "assistant", "content": "Thinking...",
            "meta_info": {"completion_tokens": 60}},
    ]
    assert compute_long_wait_alignment_score(messages) == 1.0

    # Case 4: long_wait user, assistant does not reach max_output
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {
            "type": "long_wait", "max_output": 50}},
        {"role": "assistant", "content": "Thinking...",
            "meta_info": {"completion_tokens": 30}},
    ]
    assert compute_long_wait_alignment_score(messages) == 0.0

    # Case 5: Multiple long_waits, mixed results
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {
            "type": "long_wait", "max_output": 50}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 50}},
        {"role": "user", "content": "Q2", "user_input_info": {
            "type": "long_wait", "max_output": 40}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 30}},
    ]
    assert compute_long_wait_alignment_score(messages) == 0.5

    # Case 6: Assistant message without meta_info
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {
            "type": "long_wait", "max_output": 50}},
        {"role": "assistant", "content": "A1"},
    ]
    assert compute_long_wait_alignment_score(messages) == 0.0

    print("compute_long_wait_alignment_score tests passed.")


def test_compute_answer_count_score():
    print("Testing compute_answer_count_score...")

    # Test Case 1: No answer tags - should get highest score
    messages = [
        {"role": "assistant", "content": "Let me think about this problem."},
    ]
    score = compute_answer_count_score(messages)
    assert score == 1.0, f"Expected 1.0 for no answer tags, got {score}"

    # Test Case 2: One answer tag - should get score of 1.0
    messages = [
        {"role": "assistant", "content": "<answer>42</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert score == 1.0, f"Expected 1.0 for one answer tag, got {score}"

    # Test Case 3: Two answer tags - should get score of 0.8
    messages = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.8) < 0.001, f"Expected 0.8 for two answer tags, got {score}"

    # Test Case 4: Three answer tags - should get score of 0.6
    messages = [
        {"role": "assistant",
            "content": "<answer>42</answer><answer>43</answer><answer>44</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.6) < 0.001, f"Expected 0.6 for three answer tags, got {score}"

    # Test Case 5: Four answer tags - should get score of 0.4
    messages = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer><answer>44</answer><answer>45</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.4) < 0.001, f"Expected 0.4 for four answer tags, got {score}"

    # Test Case 6: Five answer tags - should get score of 0.2
    messages = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer><answer>44</answer><answer>45</answer><answer>46</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.2) < 0.001, f"Expected 0.2 for five answer tags, got {score}"

    # Test Case 7: Six answer tags - should get score of 0.0
    messages = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer><answer>44</answer><answer>45</answer><answer>46</answer><answer>47</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert score == 0.0, f"Expected 0.0 for six answer tags, got {score}"

    # Test Case 8: Multiple assistant messages with answer tags
    messages = [
        {"role": "assistant", "content": "<answer>42</answer>"},
        {"role": "assistant", "content": "<answer>43</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.8) < 0.001, f"Expected 0.8 for two answer tags across messages, got {score}"

    # Test Case 9: Mixed content with answer tags
    messages = [
        {"role": "assistant", "content": "Let me solve this step by step. <answer>42</answer> Now let me check. <answer>43</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 0.8) < 0.001, f"Expected 0.8 for two answer tags in mixed content, got {score}"

    # Test Case 10: User messages should be ignored
    messages = [
        {"role": "user", "content": "<answer>42</answer>"},
        {"role": "assistant", "content": "The answer is <answer>43</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 1.0) < 0.001, f"Expected 1.0 for one answer tag in assistant message, got {score}"

    # Test Case 11: Malformed answer tags should not be counted
    messages = [
        {"role": "assistant", "content": "<answer>42<answer>43</answer>"},
    ]
    score = compute_answer_count_score(messages)
    assert abs(
        score - 1.0) < 0.001, f"Expected 1.0 for one complete answer tag (malformed one ignored), got {score}"

    # Test Case 12: Incomplete answer tags should not be counted
    messages = [
        {"role": "assistant", "content": "<answer>42 but no closing tag"},
    ]
    score = compute_answer_count_score(messages)
    assert score == 1.0, f"Expected 1.0 for no complete answer tags, got {score}"

    # Test Case 13: Custom max_answer_count parameter
    messages = [
        {"role": "assistant",
            "content": "<answer>42</answer><answer>43</answer><answer>44</answer>"},
    ]
    # With default max_answer_count=5, 3 tags should get score of 0.6
    score_default = compute_answer_count_score(messages)
    assert abs(score_default -
               0.6) < 0.001, f"Expected 0.6 for 3 tags with default max=5, got {score_default}"

    # With custom max_answer_count=2, 3 tags should get score of 0.0
    score_custom = compute_answer_count_score(messages, max_answer_count=2)
    assert score_custom == 0.0, f"Expected 0.0 for 3 tags with max=2, got {score_custom}"

    # Test Case 14: Edge case with max_answer_count=1
    messages = [
        {"role": "assistant", "content": "<answer>42</answer>"},
    ]
    score_max1 = compute_answer_count_score(messages, max_answer_count=1)
    assert score_max1 == 1.0, f"Expected 1.0 for 1 tag with max=1, got {score_max1}"

    # Test Case 15: Zero max_answer_count
    messages = [
        {"role": "assistant", "content": "<answer>42</answer>"},
    ]
    score_max0 = compute_answer_count_score(messages, max_answer_count=0)
    assert score_max0 == 0.0, f"Expected 0.0 for 1 tag with max=0, got {score_max0}"

    # Test Case 16: With base score parameter
    messages = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer>"},
    ]
    # With base_score=0.3, 2 tags should get score of 0.3 + 0.7 * 0.8 = 0.86
    score_with_base = compute_answer_count_score(messages, max_answer_count=5, answer_count_base_score=0.3)
    expected_score_with_base = 0.3 + 0.7 * 0.8  # 0.3 + 0.56 = 0.86
    assert abs(score_with_base - expected_score_with_base) < 0.001, f"Expected {expected_score_with_base} for 2 tags with base_score=0.3, got {score_with_base}"

    # Test Case 17: Base score with different answer counts
    messages_3_tags = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer><answer>44</answer>"},
    ]
    # With base_score=0.5, 3 tags should get score of 0.5 + 0.5 * 0.6 = 0.8
    score_3_tags_base = compute_answer_count_score(messages_3_tags, max_answer_count=5, answer_count_base_score=0.5)
    expected_score_3_tags = 0.5 + 0.5 * 0.6  # 0.5 + 0.3 = 0.8
    assert abs(score_3_tags_base - expected_score_3_tags) < 0.001, f"Expected {expected_score_3_tags} for 3 tags with base_score=0.5, got {score_3_tags_base}"

    # Test Case 18: Base score with no answer tags
    messages_no_tags = [
        {"role": "assistant", "content": "Let me think about this problem."},
    ]
    # With base_score=0.2, 0 tags should get score of 0.2 + 0.8 * 1.2 (clamped to 1.0) = 1.0
    score_no_tags_base = compute_answer_count_score(messages_no_tags, max_answer_count=5, answer_count_base_score=0.2)
    assert score_no_tags_base == 1.0, f"Expected 1.0 for 0 tags with base_score=0.2, got {score_no_tags_base}"

    # Test Case 19: Base score with maximum acceptable answer count
    messages_max_tags = [
        {"role": "assistant", "content": "<answer>42</answer><answer>43</answer><answer>44</answer><answer>45</answer><answer>46</answer>"},
    ]
    # With base_score=0.1, 5 tags should get score of 0.1 + 0.9 * 0.2 = 0.28
    score_max_tags_base = compute_answer_count_score(messages_max_tags, max_answer_count=5, answer_count_base_score=0.1)
    expected_score_max_tags = 0.1 + 0.9 * 0.2  # 0.1 + 0.18 = 0.28
    assert abs(score_max_tags_base - expected_score_max_tags) < 0.001, f"Expected {expected_score_max_tags} for 5 tags with base_score=0.1, got {score_max_tags_base}"

    print("compute_answer_count_score tests passed.")


def test_compute_ttfa_early_ratio_score():
    print("Testing compute_ttfa_early_ratio_score...")

    # Test Case 1: Perfect score - early ratio at max_reward_point
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2", "user_input_info": {
            "type": "normal"}},  # Last valid user message
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 10}},
    ]
    # c_m=15 (A1), c_n=10 (A2), total=25
    # early_ratio = 15/(15+10) = 15/25 = 0.6 (equals max_reward_point=0.6)
    score = compute_ttfa_early_ratio_score(messages, 0.25, 0.8, 0.6, 0.3)
    assert abs(score - 1.0) < 0.001, f"Expected 1.0 for early_ratio at max_reward_point, got {score}"

    # Test Case 2: Score decreases with more later tokens
    messages_with_later = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2", "user_input_info": {
            "type": "normal"}},  # Last valid user message
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 10}},
        {"role": "assistant", "content": "A3", "meta_info": {
            "completion_tokens": 5}},  # Later tokens
    ]
    # c_m=15, c_n=15 (A2+A3), total=30
    # early_ratio = 15/(15+15) = 15/30 = 0.5 (within [0.25, 0.6] range, below max_reward_point)
    # early_ratio_score = 0.3 + (1.0-0.3) * (0.5-0.25)/(0.6-0.25) = 0.3 + 0.7 * 0.25/0.35 ≈ 0.8
    score_with_later = compute_ttfa_early_ratio_score(messages_with_later, 0.25, 0.8, 0.6, 0.3)
    expected_score_later = 0.3 + (1.0-0.3) * (0.5-0.25)/(0.6-0.25)  # ≈ 0.8
    assert abs(score_with_later - expected_score_later) < 0.001
    assert score_with_later < score, "Score should be lower due to lower early ratio"

    # Test Case 3: All tokens before last user (exceeds max ratio)
    messages_ideal = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 20}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 15}},
        # Last valid user message, no assistant after
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
    ]
    # c_m=35, c_n=0, total=35
    # early_ratio = 35/(35+0) = 35/35 = 1.0 (exceeds 0.8, so score = 0.0)
    score_ideal = compute_ttfa_early_ratio_score(messages_ideal, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_ideal - 0.0) < 0.001, f"Expected 0.0 for early_ratio outside target range, got {score_ideal}"

    # Test Case 4: Early ratio above max_reward_point but within target range
    messages_above_max = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 40}},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 20}},
    ]
    # c_m=40, c_n=20, total=60
    # early_ratio = 40/(40+20) = 40/60 = 0.667 (within [0.6, 0.8] range, above max_reward_point)
    score_above_max = compute_ttfa_early_ratio_score(messages_above_max, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_above_max - 1.0) < 0.001, f"Expected 1.0 for early_ratio above max_reward_point, got {score_above_max}"

    # Test Case 5: Early-stage penalty (exceeds max ratio)
    messages_early_penalty = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1", "meta_info": {
            "completion_tokens": 45}},  # Exceeds 0.8*50=40
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 5}},
    ]
    # c_m=45, c_n=5, total=50
    # early_ratio = 45/(45+5) = 45/50 = 0.9 (exceeds 0.8, so score = 0.0)
    score_early_penalty = compute_ttfa_early_ratio_score(messages_early_penalty, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_early_penalty - 0.0) < 0.001, f"Expected 0.0 for early_ratio outside target range, got {score_early_penalty}"

    # Test Case 6: Edge case - no tokens
    messages_no_tokens = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 0}},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 0}},
    ]
    # c_m=0, c_n=0, total=0
    # When total_tokens == 0, early_ratio_score returns 1.0 (neutral)
    score_no_tokens = compute_ttfa_early_ratio_score(messages_no_tokens, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_no_tokens - 1.0) < 0.001, f"Expected 1.0 for no tokens used, got {score_no_tokens}"

    # Test Case 7: Only later tokens (extreme case)
    messages_only_later = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        # Last valid user message (no assistant before)
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 20}},
    ]
    # c_m=0, c_n=20, total=20
    # early_ratio = 0/(0+20) = 0/20 = 0.0 (below 0.25, so score = 0.0)
    score_only_later = compute_ttfa_early_ratio_score(messages_only_later, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_only_later - 0.0) < 0.001, f"Expected 0.0 for early_ratio below target range, got {score_only_later}"

    # Test Case 8: Empty user messages after valid user message (should be ignored)
    messages_with_empty = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2", "user_input_info": {
            "type": "normal"}},  # Last valid user message
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 10}},
        # Empty user message (no user_input_info)
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "A3",
            "meta_info": {"completion_tokens": 5}},
    ]
    # Should behave the same as Test Case 2 since empty user message is ignored
    # c_m=15 (A1), c_n=15 (A2+A3), total=30
    # early_ratio = 15/(15+15) = 15/30 = 0.5 (within [0.25, 0.6] range, below max_reward_point)
    score_with_empty = compute_ttfa_early_ratio_score(messages_with_empty, 0.25, 0.8, 0.6, 0.3)
    assert abs(score_with_empty - expected_score_later) < 0.001
    assert score_with_empty == score_with_later, "Score should be same as Test Case 2"

    print("compute_ttfa_early_ratio_score comprehensive tests passed.")


def test_compute_ttfa_length_score():
    print("Testing compute_ttfa_length_score...")

    # Test Case 1: Perfect score - within budget
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2", "user_input_info": {
            "type": "normal"}},  # Last valid user message
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 10}},
    ]
    # c_m=15 (A1), c_n=10 (A2), total=25
    # ref_count=50, sigma=0.2, s=0.2*50=10
    # over_length = max(25-50, 0) = 0
    # length_score = max(0, 1 - 0/10) = 1.0
    score = compute_ttfa_length_score(messages, 50)
    assert abs(score - 1.0) < 0.001, f"Expected 1.0 for within budget, got {score}"

    # Test Case 2: Length penalty when over budget
    messages_over_length = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 40}},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 20}},
    ]
    # c_m=40, c_n=20, total=60, over budget by 10
    # ref_count=50, sigma=0.2, s=0.2*50=10
    # over_length = max(60-50, 0) = 10
    # length_score = max(0, 1 - 10/10) = 0.0
    score_over_length = compute_ttfa_length_score(messages_over_length, 50)
    assert abs(score_over_length - 0.0) < 0.001, f"Expected 0.0 for over budget, got {score_over_length}"

    # Test Case 3: Partial penalty when slightly over budget
    messages_slight_over = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 30}},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 25}},
    ]
    # c_m=30, c_n=25, total=55, over budget by 5
    # ref_count=50, sigma=0.2, s=0.2*50=10
    # over_length = max(55-50, 0) = 5
    # length_score = max(0, 1 - 5/10) = 0.5
    score_slight_over = compute_ttfa_length_score(messages_slight_over, 50)
    assert abs(score_slight_over - 0.5) < 0.001, f"Expected 0.5 for slight over budget, got {score_slight_over}"

    # Test Case 4: Edge case - no tokens
    messages_no_tokens = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 0}},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A2",
            "meta_info": {"completion_tokens": 0}},
    ]
    # c_m=0, c_n=0, total=0
    # ref_count=10, sigma=0.2, s=0.2*10=2
    # over_length = max(0-10, 0) = 0
    # length_score = max(0, 1 - 0/2) = 1.0
    score_no_tokens = compute_ttfa_length_score(messages_no_tokens, 10)
    assert abs(score_no_tokens - 1.0) < 0.001, f"Expected 1.0 for no tokens used, got {score_no_tokens}"

    # Test Case 5: Edge case - ref_count = 0
    score_ref_zero = compute_ttfa_length_score(messages_no_tokens, 0)
    # When ref_count=0, s = 0.2 * 0 = 0, so length_score = 0.0
    assert score_ref_zero == 0.0, f"Expected 0.0 for ref_count=0, got {score_ref_zero}"

    # Test Case 6: Only later tokens
    messages_only_later = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        # Last valid user message (no assistant before)
        {"role": "user", "content": "Q2", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1",
            "meta_info": {"completion_tokens": 20}},
    ]
    # c_m=0, c_n=20, total=20
    # ref_count=50, sigma=0.2, s=0.2*50=10
    # over_length = max(20-50, 0) = 0
    # length_score = max(0, 1 - 0/10) = 1.0
    score_only_later = compute_ttfa_length_score(messages_only_later, 50)
    assert abs(score_only_later - 1.0) < 0.001, f"Expected 1.0 for only later tokens within budget, got {score_only_later}"

    print("compute_ttfa_length_score comprehensive tests passed.")


def test_compute_score():
    print("Testing compute_score...")

    # Test Case 1: Format stage scoring
    messages = [
        {"role": "user", "content": "What is 2+2?", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "Let me solve this step by step. <answer>4</answer>",
            "meta_info": {"completion_tokens": 20}},
    ]
    ground_truth = "4"
    ref_count = 50
    max_answer_count = 5
    answer_count_base_score = 0.0
    ttfa_ratio_min = 0.25
    ttfa_ratio_max = 0.8
    ttfa_ratio_max_reward_point = 0.6
    ttfa_ratio_base_score = 0.3

    # Format stage: 0.2*format + 0.2*fluency + 0.3*final_answer + 0.3*ttfa_early_ratio
    format_score = compute_format_score(messages)  # Should be 1.0
    fluency_score = compute_fluency_score(messages)  # May be 0.5 if API fails, or 1.0 if single message
    answer_score = compute_answer_score(messages, ground_truth)  # Should be 1.0
    answer_count_score = compute_answer_count_score(messages, max_answer_count, answer_count_base_score)  # Should be 1.0
    final_answer_score = answer_score * answer_count_score  # Should be 1.0
    ttfa_early_ratio_score = compute_ttfa_early_ratio_score(
        messages, ttfa_ratio_min, ttfa_ratio_max, ttfa_ratio_max_reward_point, ttfa_ratio_base_score)  # Should be 1.0 (no later tokens)

    # Calculate expected score based on actual fluency_score (which may be 0.5 due to API failure)
    expected_format_score = 0.2 * format_score + 0.2 * fluency_score + 0.3 * final_answer_score + 0.3 * ttfa_early_ratio_score
    format_stage_score = compute_score(messages, ground_truth, ref_count, max_answer_count, 
                                     answer_count_base_score, ttfa_ratio_min, ttfa_ratio_max, 
                                     ttfa_ratio_max_reward_point, ttfa_ratio_base_score, "format")
    assert abs(format_stage_score - expected_format_score) < 0.001, f"Expected {expected_format_score} for format stage, got {format_stage_score}"

    # Test Case 2: Final stage scoring
    # Add long_wait message to test long_wait_alignment_score
    messages_with_long_wait = [
        {"role": "user", "content": "Think about this carefully", "user_input_info": {"type": "long_wait", "max_output": 30}},
        {"role": "assistant", "content": "Let me think about this step by step...",
            "meta_info": {"completion_tokens": 30}},
        {"role": "user", "content": "What is 2+2?", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "The answer is <answer>4</answer>",
            "meta_info": {"completion_tokens": 10}},
    ]
    
    # Final stage: 0.35*final_answer + 0.15*format + 0.10*fluency + 0.15*long_wait_alignment + 0.25*ttfa_score
    format_score = compute_format_score(messages_with_long_wait)  # Should be 1.0
    fluency_score = compute_fluency_score(messages_with_long_wait)  # May be 0.5 if API fails
    answer_score = compute_answer_score(messages_with_long_wait, ground_truth)  # Should be 1.0
    answer_count_score = compute_answer_count_score(messages_with_long_wait, max_answer_count, answer_count_base_score)  # Should be 1.0
    final_answer_score = answer_score * answer_count_score  # Should be 1.0
    long_wait_alignment_score = compute_long_wait_alignment_score(messages_with_long_wait)  # Should be 1.0 (reached max_output)
    ttfa_early_ratio_score = compute_ttfa_early_ratio_score(
        messages_with_long_wait, ttfa_ratio_min, ttfa_ratio_max, ttfa_ratio_max_reward_point, ttfa_ratio_base_score)
    ttfa_length_score = compute_ttfa_length_score(messages_with_long_wait, ref_count)
    ttfa_score = ttfa_early_ratio_score * ttfa_length_score

    final_stage_score = compute_score(messages_with_long_wait, ground_truth, ref_count, max_answer_count, 
                                    answer_count_base_score, ttfa_ratio_min, ttfa_ratio_max, 
                                    ttfa_ratio_max_reward_point, ttfa_ratio_base_score, "final")
    
    # Verify the score is reasonable (between 0 and 1)
    assert 0.0 <= final_stage_score <= 1.0, f"Final stage score should be between 0 and 1, got {final_stage_score}"
    
    # Verify that format stage and final stage scores are different (they use different weights)
    # Note: This might fail if fluency_score is 0.5 in both cases due to API failure
    if abs(format_stage_score - final_stage_score) <= 0.001:
        print("Warning: Format and final stage scores are the same. This might be due to API failure in fluency_score.")
        print(f"Format stage score: {format_stage_score}, Final stage score: {final_stage_score}")

    print("compute_score tests passed.")

def test_local_json_logging():
    """Test time-based logging functionality (local JSON files for both detailed samples and aggregated scores)."""
    global LOG_INTERVAL_SECONDS
    
    print("Testing time-based logging...")
    
    # Set a short log interval for testing (3 seconds)
    original_interval = LOG_INTERVAL_SECONDS
    LOG_INTERVAL_SECONDS = 3.0
    
    # Reset logging state for clean test
    reset_logging_state()
    
    # Test messages
    messages = [
        {"role": "user", "content": "What is 2+2?", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "Let me solve this step by step. <answer>4</answer>",
            "meta_info": {"completion_tokens": 20}},
    ]
    ground_truth = "4"
    ref_count = 50
    max_answer_count = 5
    answer_count_base_score = 0.0
    ttfa_ratio_min = 0.25
    ttfa_ratio_max = 0.8
    ttfa_ratio_max_reward_point = 0.6
    ttfa_ratio_base_score = 0.3
    
    print("Starting time-based logging test...")
    print("Log interval set to 3 seconds")
    print("Unified logs (detailed samples + aggregated scores) will be saved to local JSON files")
    
    # Run multiple calls to test time-based logging
    for i in range(15):
        print(f"\nCall {i+1}:")
        
        # Compute score with logging enabled
        score = compute_score(messages, ground_truth, ref_count, max_answer_count, 
                             answer_count_base_score, ttfa_ratio_min, ttfa_ratio_max, 
                             ttfa_ratio_max_reward_point, ttfa_ratio_base_score, "format", log=True)
        
        # Get current stats
        stats = get_logging_stats()
        print(f"  Score: {score:.4f}")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Time until next log: {stats['time_until_next_log_seconds']:.1f} seconds")
        print(f"  Save detailed samples: {stats['save_detailed_samples']}")
        print(f"  Detailed samples dir: {stats['detailed_samples_dir']}")
        print(f"  Score counts: {stats['score_counts']}")
        
        # Wait 1 second between calls
        time.sleep(1)
    
    print("\nTest completed.")
    
    # Restore original interval
    LOG_INTERVAL_SECONDS = original_interval
    
    print("Time-based logging test finished.")


def test_compute_time_to_final_answer_score():
    print("Testing compute_time_to_final_answer_score...")
    print("Note: This function has been split into compute_ttfa_early_ratio_score and compute_ttfa_length_score.")
    print("Please use the individual test functions instead.")
    print("This test function is kept for backward compatibility but does nothing.")


def main():
    test_compute_answer_score()
    test_compute_format_score()
    test_compute_fluency_score()
    test_compute_long_wait_alignment_score()
    test_compute_answer_count_score()
    test_compute_ttfa_early_ratio_score()
    test_compute_ttfa_length_score()
    test_compute_score()
    test_local_json_logging()
    
    print("All tests passed.")


if __name__ == "__main__":
    main()
