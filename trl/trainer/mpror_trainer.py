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
import typing
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)
from ..models import PreTrainedModelWrapper

from trl.trainer.mpror_config import MPRORConfig
from trl.trainer.ppo_trainer import PPOTrainer



MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class MPRORTrainer(PPOTrainer):
    """
    The MPRORTraining uses Multiple Policy Rollouts per Oracle Rollin to optimise language models.
    Note, this trainer is a modified version of PPOTrainer. 

    Attributes:
        **config** (`MPRORConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: MPRORConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize MPRORTrainer.

        Args:
            config (`MPRORConfig`):
                Configuration object for PPOTrainer. Check the documentation of `MPRORConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        config.batch_size = config.batch_size * config.max_num_rollouts
        config.gradient_accumulation_steps = config.gradient_accumulation_steps * config.max_num_rollouts
        config.horizon = config.horizon * config.max_num_rollouts
        super().__init__(config, 
                        model,
                        ref_model,
                        tokenizer,
                        dataset,
                        optimizer,
                        data_collator,
                        num_shared_layers,
                        lr_scheduler)


    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        assert "output_ids" in dataset[0], "Dataset must contain `output_ids`"

        def prepare_for_rollout(batch):
            """
            Prepare for rollouts by expanding input prompt `input_ids` to various 
            lengths based on `interval` and `max_num_rollouts` in the config.

            We will then do rollouts at each of these rolled-in inputs. 

            Args:
                batch (`dict`): Batch of data

            Returns:
                `dict`: Batch of data
            """
            new_batch = {}
            
            batch['num_rollouts'] = []

            for input_ids, output_ids in zip(batch['input_ids'], batch['output_ids']):
                num_rollouts = min(self.config.max_num_rollouts, 
                                    len(output_ids) // self.config.interval)
                batch['num_rollouts'].append(num_rollouts)

            for key, values in batch.items():
                if key == 'num_rollouts':
                    continue

                new_batch[key] = []

                for i, value in enumerate(values):

                    num_rollouts = batch['num_rollouts'][i]
                    if key == "input_ids":
                        input_ids = value
                        output_ids = batch['output_ids'][i]
                        max_rollin = min(len(output_ids),self.config.max_rollout_length) - len(input_ids)
                        
                        if not self.config.exclude_first:
                            new_batch[key].append(input_ids)
                            num_rollouts -= 1
                        if not self.config.exclude_last:
                            new_batch[key].append(
                                torch.cat((input_ids, output_ids), dim=0))
                            num_rollouts -= 1

                        rollin_intervals = range(0, max_rollin, self.config.interval)
                        for l in sorted(np.random.choice(rollin_intervals, 
                                            size=num_rollouts, replace=False)):
                            new_batch[key].append(
                                torch.cat((input_ids, output_ids[:l]), dim=0))
                    else:    
                        new_batch[key].extend([value] * num_rollouts)
            return new_batch
            
        dataset = dataset.map(
            prepare_for_rollout,
            batched=True,
            batch_size=self.config.mini_batch_size,
        )
        dataset.set_format(type="torch")
    
        return super().prepare_dataloader(
                        dataset, data_collator=data_collator)

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = None,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if generation_kwargs is None:
            generation_kwargs = {}

        if batch_size is None:
            batch_size = self.config.mini_batch_size
        
        if isinstance(query_tensor, List):
            return super().generate(
                query_tensor=query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,)
        
        responses = []    
        # Here the input would be of the size self.config.batch_size 
        # which for mpror-rl is actual_batch_size * max_num_rollouts. 
        # Hence, we also do generation with mini_batch_size.
        for i in range(0, len(query_tensor), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensor), i + batch_size)
            batch = query_tensor[i:end_index]

            response = super().generate(
                batch,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            responses.append(response)
        return torch.cat(responses, dim=0)