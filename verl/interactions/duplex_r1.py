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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score.duplex import duplex_r1

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DuplexR1Interaction(BaseInteraction):
    """A demo interaction for calculating the reward of gsm8k.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        print(f"DuplexR1Interaction config: {config}")
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str], current_turn_data: dict, **interaction_kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = interaction_kwargs
        user_input_info = self._instance_dict[instance_id]["user_input_info"]
        current_turn_data["request_sampling_params"]["max_new_tokens"] = user_input_info[0]["max_output"]
        return instance_id

    async def generate_response(
        self, instance_id: str, current_turn_data: dict, messages: List[Dict[str, Any]], **_interaction_kwargs
    ) -> Tuple[bool, str, float, dict]:
        interaction_kwargs = self._instance_dict[instance_id]
        user_input_info = interaction_kwargs["user_input_info"]
        messages = duplex_r1.append_user_input_info(messages, user_input_info)
        
        user_turns = current_turn_data["user_turns"]

        answer_count_score = duplex_r1.compute_answer_count_score(messages, interaction_kwargs["max_answer_count"])
        answer_score = duplex_r1.compute_answer_score(messages, interaction_kwargs["ground_truth"])

        if answer_count_score == 0.0 or answer_score == 1.0:
            # Reached the maximum answer count or the answer is correct
            response = ""
            should_terminate_sequence = True
        elif self._instance_dict[instance_id].get("is_last_round", False):
            response = ""
            should_terminate_sequence = True
        elif user_turns >= interaction_kwargs["max_user_round"] - 1:
            # Reached the maximum user round
            response = "You have reached the maximum rounds of reasoning. You MUST output your final answer NOW using the format: <answer>YOUR_MATH_ANSWER_WITHOUT_ANY_OTHER_TEXT</answer>."
            self._instance_dict[instance_id]["is_last_round"] = True
            should_terminate_sequence = False
        elif user_turns < len(user_input_info):
            # set `max_new_tokens` for the next assistant generation
            current_turn_data["request_sampling_params"]["max_new_tokens"] = user_input_info[user_turns]["max_output"]
            response = user_input_info[user_turns]["input_text"]
            should_terminate_sequence = False
        else:
            response = ""
            should_terminate_sequence = False

        return should_terminate_sequence, response, 0.0, {}

    async def calculate_score(self, **kwargs) -> float:
        return duplex_r1.compute_score(**kwargs)

    async def finalize_interaction(self, instance_id: str) -> None:
        del self._instance_dict[instance_id]
