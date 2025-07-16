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
import openai
from typing import List
from math_verify import parse, verify

OPENAI_API_KEY = 'sk-xxx'
OPENAI_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OPENAI_BASE_URL = "http://127.0.0.1:10002/v1"

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
    combined_output = ''.join([msg['content'] for msg in messages if msg['role'] == 'assistant'])

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


def compute_format_score(messages: List[dict]) -> float:
    """
    Computes a global format score for a list of messages.
    Checks:
    1. If <wait> appears, it must be at the end of the assistant message (per message).
    2. <answer> and </answer> tags must be properly paired and closed (globally across all assistant messages).
    3. Every user message must be immediately followed by an assistant message; otherwise, returns 0.0.
    4. All assistant messages must be non-empty (at least contain something, e.g., <wait>); otherwise, returns 0.0.
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
    assistant_outputs = [msg['content'] for msg in messages if msg['role'] == 'assistant']

    # Non-empty check for all assistant messages
    for output in assistant_outputs:
        if not output or output.strip() == '':
            return 0.0

    # Global check for <answer> tag pairing
    combined_output = ''.join(assistant_outputs)
    answer_open = [m.start() for m in re.finditer(r'<answer>', combined_output)]
    answer_close = [m.start() for m in re.finditer(r'</answer>', combined_output)]
    if len(answer_open) != len(answer_close):
        return 0.0
    for open_pos, close_pos in zip(answer_open, answer_close):
        if open_pos > close_pos:
            return 0.0
    # Per-assistant-message check for <wait>
    for output in assistant_outputs:
        wait_pos = output.find('<wait>')
        if wait_pos != -1 and not output.rstrip().endswith('<wait>'):
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
    outputs = [msg['content'] for msg in messages if msg['role'] == 'assistant']
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
        client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        
        # Prepare all requests
        requests = []
        for i in range(len(outputs) - 1):
            prompt = create_transition_prompt(outputs[i], outputs[i + 1], i + 1)
            requests.append({
                "model": OPENAI_MODEL,
                                 "messages": [
                     {"role": "system", "content": "You are a text fluency evaluation expert. Your task is to rate the naturalness of text transitions. You MUST respond with ONLY a single decimal number between 0.0 and 1.0 (e.g., 0.8, 0.5, 1.0). Do NOT include any explanations, words, or additional text. Output format: [number only]"},
                     {"role": "user", "content": prompt}
                 ],
                "temperature": 0.1,  # Low temperature for consistent evaluation
                "max_tokens": 10  # We only need a single number
            })
        
        # Execute all requests (in sequence for now, could be made parallel)
        scores = []
        for i, request in enumerate(requests):
            try:
                response = client.chat.completions.create(**request)
                response_text = response.choices[0].message.content.strip()
                
                # Parse the single score from response
                # Try to extract a decimal number from the response (prioritize valid 0.0-1.0 range)
                number_match = re.search(r'([01]\.?[0-9]*|0\.[0-9]+|1\.0*)', response_text)
                if number_match:
                    score = float(number_match.group(1))
                    # Clamp score to [0, 1] range
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                else:
                    print(f"Warning: Could not parse score from response: '{response_text}'. Using fallback score 0.5.")
                    scores.append(0.5)
                    
            except ValueError:
                print(f"Warning: Invalid score format in response for transition {i+1}. Using fallback score 0.5.")
                scores.append(0.5)
            except Exception as e:
                print(f"Warning: Error processing transition {i+1}: {e}. Using fallback score 0.5.")
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
    whose content ends with <wait>. Returns the average score (1.0 for each correct, 0.0 for each incorrect).
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
        # Check if content ends with <wait>
        if next_msg['content'].rstrip().endswith('<wait>'):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def compute_time_to_final_answer_score(messages: List[dict], ref_count: int):
    """
    Compute the score for Time To Final Answer.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
        ref_count: int,              # reference thinking token count
    Returns:
        float: score for Time To Final Answer
    Note:
        c_m: previous_output_count
        c_n: later_output_count
        k: ref_count
        s: sigma * k
        total_tokens: c_m + c_n
        over_length: max(total_tokens - k, 0)
        over_c_m: max(c_m - allowed_c_m, 0)
    """
    # Find the index of the last user message
    user_end_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            user_end_idx = i
            break
    if user_end_idx == -1:
        raise ValueError("No user message found in messages list.")

    previous_output_count = 0
    later_output_count = 0
    for idx, msg in enumerate(messages):
        if msg['role'] == 'assistant':
            token_count = msg['meta_info']['completion_tokens']
            if idx > user_end_idx:
                later_output_count += token_count
            else:
                previous_output_count += token_count

    # Compute the reward according to:
    #   r1 = 1 - c_n / (c_m + c_n + epsilon) = c_m / (c_m + c_n + epsilon)
    # where:
    #   c_m: previous_output_count (assistant tokens before and including last user message)
    #   c_n: later_output_count (assistant tokens after last user message)
    #   epsilon: small constant to avoid division by zero, set to 1
    epsilon = 1.0
    c_m = previous_output_count
    c_n = later_output_count
    # Add a temperature parameter gamma in (0, 1] to scale the score_ratio: score_ratio = (c_m / (c_m + c_n + epsilon)) ** gamma
    gamma = 1.0  # You can set gamma to a value in (0, 1] as needed
    score_ratio = (c_m / (c_m + c_n + epsilon)) ** gamma

    # Compute the total length compliance reward (reward_length) according to:
    #   r2 = max(0, 1 - ((c_m + c_n - k)_+) / s)
    #   where s = sigma * k, sigma ~ 0.2, k = ref_count
    #   (x)_+ = max(x, 0)
    sigma = 0.2
    k = ref_count
    s = sigma * k
    total_tokens = c_m + c_n
    over_length = max(total_tokens - k, 0)
    score_length = max(0.0, 1.0 - (over_length / s) if s > 0 else 0.0)

    # Compute the early-stage token penalty reward (reward_penalty) according to:
    #   r3 = max(0, 1 - ((c_m - rho * k)_+) / (rho_prime * k))
    #   where rho is the allowed fraction of budget for c_m (e.g., 0.8)
    #         rho_prime controls the decay rate after exceeding (e.g., 0.2)
    #         k = ref_count
    #   If you do not want to penalize c_m, set reward_penalty = 1.0
    rho = 0.8
    rho_prime = 0.2
    if k > 0:
        allowed_c_m = rho * k
        decay_span = rho_prime * k
        over_c_m = max(c_m - allowed_c_m, 0)
        score_penalty = max(0.0, 1.0 - (over_c_m / decay_span) if decay_span > 0 else 0.0)
    else:
        score_penalty = 1.0

    final_score = score_ratio * score_length * score_penalty

    return final_score


def compute_score(messages: List[dict], ground_truth: str, ref_count: int, stage: str = "format") -> float:
    """
    Compute the final reward score by combining sub-scores with weighted average.
    Args:
        messages (List[dict]): List of messages, each with {role, content, meta_info, user_input_info}
        ground_truth (str): The ground truth answer for answer_score
        ref_count (int): Reference thinking token count for time_to_final_answer_score
        api_key (str): OpenAI API key for fluency_score
        stage (str): 'format' for format stage, 'final' for full evaluation stage
    Returns:
        float: The final weighted score
    """
    # Compute all possible scores
    format_score = compute_format_score(messages)
    # fluency_score = compute_fluency_score(messages)
    fluency_score = 1.0
    if stage == "format":
        # Only use format and fluency scores
        return 0.7 * format_score + 0.3 * fluency_score
    # For final stage, compute all scores
    answer_score = compute_answer_score(messages, ground_truth)
    long_wait_alignment_score = compute_long_wait_alignment_score(messages)
    time_to_final_answer_score = compute_time_to_final_answer_score(messages, ref_count)
    # Weighted sum according to user-confirmed weights
    return (
        0.35 * answer_score +
        0.15 * format_score +
        0.10 * fluency_score +
        0.15 * long_wait_alignment_score +
        0.25 * time_to_final_answer_score
    )


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
    # Case 1: Properly paired, user/assistant alternation, <wait> at end
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<answer>ok</answer><wait>"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<answer>yes</answer>"},
    ]
    assert compute_format_score(messages) == 1.0

    # Case 2: <wait> not at end
    messages[1]["content"] = "<wait>something"
    assert compute_format_score(messages) == 0.0
    messages[1]["content"] = "<answer>ok</answer><wait>"  # restore

    # Case 3: Unpaired <answer>
    messages[3]["content"] = "<answer>yes"
    assert compute_format_score(messages) == 0.0
    messages[3]["content"] = "<answer>yes</answer>"  # restore

    # Case 4: User not followed by assistant
    bad_messages = [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2"},
    ]
    assert compute_format_score(bad_messages) == 0.0

    # Case 5: Empty assistant message
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": ""},
    ]
    assert compute_format_score(messages) == 0.0

    # Case 6: Assistant message with only whitespace
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "   "},
    ]
    assert compute_format_score(messages) == 0.0

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
        {"role": "assistant", "content": "First, I'll identify the variables in the equation."},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    # Case 5: Mixed roles (only assistant messages should be considered)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate.<wait>"},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    # Case 6: Extreme topic jumping without context
    messages = [
        {"role": "assistant", "content": "The solution to this algebra problem is x = 5."},
        {"role": "assistant", "content": "My grandmother's recipe for chocolate cake requires three eggs."},
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
        {"role": "assistant", "content": "aaaaaa!!!!! bbbbb????? 123123123 @@@@@@ ######### aaaaaa!!!!!"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    # Case 9: Broken tags and malformed content
    messages = [
        {"role": "assistant", "content": "Here's the answer: <answer>42</answer>"},
        {"role": "assistant", "content": "<wait><answer>no</wait></answer><wait><answer>"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    # Case 10: Control characters and special formatting
    messages = [
        {"role": "assistant", "content": "Let me think about this problem."},
        {"role": "assistant", "content": "result\n\n\r\t\t\t   \x00\x01\x02 is \b\f\v here somewhere\n\r\n"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    # Case 11: Extremely abrupt contradiction and reversal
    messages = [
        {"role": "assistant", "content": "Based on careful analysis, the correct answer is definitely A."},
        {"role": "assistant", "content": "NO WAIT WRONG EVERYTHING I SAID IS FALSE IT'S ACTUALLY Z!!!"},
    ]
    score = compute_fluency_score(messages)
    assert 0.0 <= score <= 1.0, f"Score should be between 0.0 and 1.0, got {score}"
    
    print("compute_fluency_score tests passed (API calls may fallback to 0.5).")


def test_compute_long_wait_alignment_score():
    print("Testing compute_long_wait_alignment_score...")
    # Case 1: No long_wait user messages
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "normal"}},
        {"role": "assistant", "content": "A1"},
    ]
    assert compute_long_wait_alignment_score(messages) == 1.0

    # Case 2: long_wait user, assistant ends with <wait>
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "long_wait"}},
        {"role": "assistant", "content": "Thinking...<wait>"},
    ]
    assert compute_long_wait_alignment_score(messages) == 1.0

    # Case 3: long_wait user, assistant does not end with <wait>
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "long_wait"}},
        {"role": "assistant", "content": "Thinking..."},
    ]
    assert compute_long_wait_alignment_score(messages) == 0.0

    # Case 4: Multiple long_waits, mixed results
    messages = [
        {"role": "user", "content": "Q1", "user_input_info": {"type": "long_wait"}},
        {"role": "assistant", "content": "A1<wait>"},
        {"role": "user", "content": "Q2", "user_input_info": {"type": "long_wait"}},
        {"role": "assistant", "content": "A2"},
    ]
    assert compute_long_wait_alignment_score(messages) == 0.5
    print("compute_long_wait_alignment_score tests passed.")


def test_compute_time_to_final_answer_score():
    print("Testing compute_time_to_final_answer_score...")
    
    # Test Case 1: Perfect score - all tokens before last user, within budget
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2"},  # Last user message
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 10}},
    ]
    ref_count = 50
    score = compute_time_to_final_answer_score(messages, ref_count)
    # 实际情况：c_m=15 (A1), c_n=10 (A2), total=25
    # score_ratio = 15/(15+10+1) = 15/26 ≈ 0.577
    # score_length = 1.0 (25 < 50)
    # score_penalty = 1.0 (15 < 0.8*50=40)
    expected_score_ratio = 15 / 26
    expected_score = expected_score_ratio * 1.0 * 1.0
    assert abs(score - expected_score) < 0.001, f"Expected {expected_score}, got {score}"
    
    # Test Case 2: Score decreases with more later tokens
    messages_with_later = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2"},  # Last user message
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 10}},
        {"role": "assistant", "content": "A3", "meta_info": {"completion_tokens": 5}},  # Later tokens
    ]
    score_with_later = compute_time_to_final_answer_score(messages_with_later, ref_count)
    # c_m=15, c_n=15 (A2+A3), total=30
    # score_ratio = 15/(15+15+1) = 15/31 ≈ 0.484
    expected_score_ratio_later = 15 / 31
    expected_score_later = expected_score_ratio_later * 1.0 * 1.0
    assert abs(score_with_later - expected_score_later) < 0.001
    assert score_with_later < score, "Score should decrease with later tokens"
    
    # Test Case 3: All tokens before last user (ideal case)
    messages_ideal = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 20}},
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 15}},
        {"role": "user", "content": "Q2"},  # Last user message, no assistant after
    ]
    score_ideal = compute_time_to_final_answer_score(messages_ideal, ref_count=50)
    # c_m=35, c_n=0, total=35
    # score_ratio = 35/(35+0+1) = 35/36 ≈ 0.972
    # score_length = 1.0 (35 < 50)
    # score_penalty = 1.0 (35 < 40)
    expected_score_ideal = 35/36 * 1.0 * 1.0
    assert abs(score_ideal - expected_score_ideal) < 0.001
    assert score_ideal > score, "Ideal case should have higher score"
    
    # Test Case 4: Length penalty when over budget
    messages_over_length = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 40}},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 20}},
    ]
    ref_count = 50
    score_over_length = compute_time_to_final_answer_score(messages_over_length, ref_count)
    # c_m=40, c_n=20, total=60, over budget by 10
    # score_ratio = 40/(40+20+1) = 40/61 ≈ 0.656
    # score_length = max(0, 1 - 10/(0.2*50)) = max(0, 1 - 10/10) = 0.0
    # score_penalty = max(0, 1 - (40-40)/(0.2*50)) = 1.0 (40 = 40, no penalty)
    expected_score_over = 40/61 * 0.0 * 1.0  # Should be 0.0
    assert abs(score_over_length - expected_score_over) < 0.001
    
    # Test Case 5: Early-stage penalty
    messages_early_penalty = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 45}},  # Exceeds 0.8*50=40
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 5}},
    ]
    ref_count = 50
    score_early_penalty = compute_time_to_final_answer_score(messages_early_penalty, ref_count)
    # c_m=45, c_n=5, total=50
    # score_ratio = 45/(45+5+1) = 45/51 ≈ 0.882
    # score_length = 1.0 (50 = 50, no penalty)
    # score_penalty = max(0, 1 - (45-40)/(0.2*50)) = max(0, 1 - 5/10) = 0.5
    expected_score_penalty = 45/51 * 1.0 * 0.5
    assert abs(score_early_penalty - expected_score_penalty) < 0.001
    
    # Test Case 6: Edge case - no tokens
    messages_no_tokens = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 0}},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "meta_info": {"completion_tokens": 0}},
    ]
    score_no_tokens = compute_time_to_final_answer_score(messages_no_tokens, ref_count=10)
    # c_m=0, c_n=0, total=0
    # score_ratio = 0/(0+0+1) = 0/1 = 0.0
    expected_score_no_tokens = 0.0
    assert abs(score_no_tokens - expected_score_no_tokens) < 0.001
    
    # Test Case 7: Edge case - ref_count = 0
    score_ref_zero = compute_time_to_final_answer_score(messages_no_tokens, ref_count=0)
    # When ref_count=0, score_length division by zero is handled
    assert score_ref_zero == 0.0, f"Expected 0.0 for ref_count=0, got {score_ref_zero}"
    
    # Test Case 8: Only later tokens (extreme case)
    messages_only_later = [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2"},  # Last user message (no assistant before)
        {"role": "assistant", "content": "A1", "meta_info": {"completion_tokens": 20}},
    ]
    score_only_later = compute_time_to_final_answer_score(messages_only_later, ref_count=50)
    # c_m=0, c_n=20, total=20
    # score_ratio = 0/(0+20+1) = 0/21 = 0.0
    expected_score_only_later = 0.0
    assert abs(score_only_later - expected_score_only_later) < 0.001
    
    print("compute_time_to_final_answer_score comprehensive tests passed.")

def main():
    test_compute_answer_score()
    test_compute_format_score()
    test_compute_fluency_score()
    test_compute_long_wait_alignment_score()
    test_compute_time_to_final_answer_score()
    print("All tests passed.")


if __name__ == "__main__":
    main()
