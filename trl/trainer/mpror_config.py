# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Optional

from trl.trainer.ppo_config import PPOConfig

@dataclass
class MPRORConfig(PPOConfig):
    """
    Configuration class for PPOTrainer
    """
    interval: Optional[int] = field(
        default=1,
        metadata={"help": "Do rollouts every interval steps."},
    )
    max_num_rollouts: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of rollouts to do per instance."},
    )
    max_rollout_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum length of rollout trajectories."},
    )
    exclude_first: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at first step (t=0)."},
    )
    exclude_last: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    def __post_init__(self):
        super().__post_init__()
