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
        self, instance_id: str, current_turn_data: dict, messages: List[Dict[str, Any]], **interaction_kwargs
    ) -> Tuple[bool, str, float, dict]:
        user_input_info = self._instance_dict[instance_id]["user_input_info"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i < len(user_input_info):
                msg["user_input_info"] = user_input_info[i]
        
        user_turns = current_turn_data["user_turns"]
        
        if user_turns < len(user_input_info):
            # set `max_new_tokens` for the next assistant generation
            current_turn_data["request_sampling_params"]["max_new_tokens"] = user_input_info[user_turns]["max_output"]
            response = user_input_info[user_turns]["input_text"]
            should_terminate_sequence = False
        else:
            response = ""
            should_terminate_sequence = True

        reward = await self.calculate_score(instance_id, messages)

        return should_terminate_sequence, response, reward, {}

    async def calculate_score(self, instance_id: str, messages: List[Dict[str, Any]]) -> float:
        return duplex_r1.compute_score(
            messages,
            self._instance_dict[instance_id]["ground_truth"],
            self._instance_dict[instance_id]["generations_max_token"]
        )

    async def finalize_interaction(self, instance_id: str) -> None:
        del self._instance_dict[instance_id]
