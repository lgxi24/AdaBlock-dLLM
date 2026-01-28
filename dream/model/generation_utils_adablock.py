# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Copyright 2025 Guanxi Lu, Imperial College London (modifications)
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream
# Modified by Guanxi Lu, Imperial College London

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    nfe_history: Optional[list] = None
    block_history: Optional[list] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", 8)
        dual_cache = kwargs.get("dual_cache", False)
        delimiter_threshold = kwargs.get("delimiter_threshold", 0.3)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            delimiter_threshold=delimiter_threshold
        )
        return result

    def _compute_block_length(self, logits, prompt_length, gen_length, generated_length, default_block_length, delimiter_threshold=0.3):
        """
        Compute adaptive block length based on delimiter confidence.
        Returns the position of the highest-confidence delimiter if above threshold,
        otherwise returns default_block_length.
        """
        block_start = prompt_length + generated_length
        remaining_length = gen_length - generated_length
        
        # Create sampling window (25% of gen_length, capped by remaining)
        window_size = min(int(0.25 * gen_length), remaining_length)
        window_positions = torch.arange(block_start, block_start + window_size, device=logits.device)
        
        # Get predicted tokens in window
        window_tokens = torch.argmax(logits[0, window_positions, :], dim=-1)
        
        # Delimiter tokens for Dream tokenizer
        delimiter_ids = [198, 271, 280, 319, 340, 382, 401, 532, 624, 630, 692, 698, 921, 1248, 1837, 1939, 2219, 2533, 3276, 3876, 4894, 5267, 14750, 68327]
        
        # Create mask for delimiter tokens
        delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
        for token_id in delimiter_ids:
            delimiter_mask |= (window_tokens == token_id)
        
        # Fallback to default block length if no delimiter is found
        if not torch.any(delimiter_mask):
            return min(default_block_length, remaining_length)
        
        positions_rel = torch.nonzero(delimiter_mask).squeeze(-1)
        delimiter_pos = window_positions[positions_rel]
        matched_tokens = window_tokens[positions_rel]
        delimiter_confidences = []
        for i, pos in enumerate(delimiter_pos):
            token_id = matched_tokens[i].item()
            delimiter_logit = logits[0, pos, token_id]
            log_sum_exp = torch.logsumexp(logits[0, pos, :], dim=-1)
            delimiter_confidence = torch.exp(delimiter_logit - log_sum_exp)
            delimiter_confidences.append(delimiter_confidence)
        
        delimiter_confidences = torch.stack(delimiter_confidences)
        max_confidence, best_idx = torch.max(delimiter_confidences, dim=0)
        max_confidence = max_confidence.item()
        best_delimiter_pos = delimiter_pos[best_idx].item()
        
        if max_confidence >= delimiter_threshold:
            block_length = best_delimiter_pos - block_start + 1
        else:
            block_length = min(default_block_length, remaining_length)
        return block_length


    def _sample_adablock(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,  # init_block_length: default block length for adaptive block sizing
        dual_cache: bool = False,
        delimiter_threshold: Optional[float] = 0.3,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        AdaBlock-dLLM Implementation without Cache
        """

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        def _confidence_threshold_sample(
            logits: torch.Tensor,
            mask_index: torch.Tensor,
            x: torch.Tensor,
            block_start: int,
            block_end: int,
            mask_token_id: int,
            temperature: float,
            top_p: Optional[float],
            top_k: Optional[int],
            threshold: float,
        ) -> torch.Tensor:
            mask_logits = logits[mask_index]
        
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            
            x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            
            x_[mask_index] = x0.clone()
            full_confidence[mask_index] = confidence
            full_confidence[:, block_end:] = -torch.inf
            
            current_transfer_tokens = (x[:, block_start:block_end] == mask_token_id).sum()
            
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
            
            select_index = select_index.to(x.device)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < threshold:
                    transfer_index[0, select_index[0, k]] = False
            
            x[transfer_index] = x_[transfer_index]
            return x

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        prompt_length = input_ids.shape[1]
        generated_length = 0
        nfe_history = []
        block_history = []
        # Process each block
        while generated_length < gen_length:
            nfe = 0
            block_start = prompt_length + generated_length

            model_output = self(x, attention_mask, tok_idx if tok_idx is not None else None)
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            nfe += 1

            # block length prediction
            current_block_length = self._compute_block_length(logits, prompt_length, gen_length, generated_length, block_length, delimiter_threshold=delimiter_threshold)
            block_history.append(current_block_length)
            generated_length += current_block_length
            block_end = block_start + current_block_length

            # denoise
            mask_index = (x == mask_token_id)
            mask_index[:, block_end:] = False

            # prepare attention mask
            if attention_mask != "full":
                current_attention_mask = attention_mask[:, :, :, block_start:]
            else:
                current_attention_mask = attention_mask

            if alg == 'confidence_threshold':
                x = _confidence_threshold_sample(
                    logits=logits,
                    mask_index=mask_index,
                    x=x,
                    block_start=block_start,
                    block_end=block_end,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=threshold
                )
            else:
                raise NotImplementedError(alg)

            while True:
                if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                    break

                # denoise
                mask_index = (x == mask_token_id)
                mask_index[:, block_end:] = False
                
                # prepare attention mask
                if attention_mask != "full":
                    # adjust attention mask
                    current_attention_mask = attention_mask[:, :, :, block_start:]
                else:
                    current_attention_mask = attention_mask

                model_output = self(x, current_attention_mask, tok_idx if tok_idx is not None else None)

                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                nfe += 1

                # sample
                if alg == 'confidence_threshold':
                    x = _confidence_threshold_sample(
                        logits=logits,
                        mask_index=mask_index,
                        x=x,
                        block_start=block_start,
                        block_end=block_end,
                        mask_token_id=mask_token_id,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        threshold=threshold
                    )
                else: 
                    raise NotImplementedError(alg)
            
            nfe_history.append(nfe)
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
                block_history=block_history
            )
        else:
            return x

    def _sample_adablock_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,  # init_block_length: default block length for adaptive block sizing
        dual_cache: bool = False,
        delimiter_threshold: Optional[float] = 0.3,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        AdaBlock-dLLM Implementation with Dual Cache
        """

        # prefix cache is not supported
        assert dual_cache, "Only dual cache is supported"

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        def _confidence_threshold_sample(
            logits: torch.Tensor,
            mask_index: torch.Tensor,
            x: torch.Tensor,
            block_start: int,
            block_end: int,
            mask_token_id: int,
            temperature: float,
            top_p: Optional[float],
            top_k: Optional[int],
            threshold: float,
        ) -> torch.Tensor:
            mask_logits = logits[mask_index]
        
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            
            x_ = torch.zeros_like(x[:, block_start:block_end], device=self.device, dtype=torch.long) + mask_token_id
            full_confidence = torch.full_like(x[:, block_start:block_end], -torch.inf, device=self.device, dtype=logits.dtype)
            
            x_[mask_index] = x0.clone()
            full_confidence[mask_index] = confidence
            full_confidence[:, block_end:] = -torch.inf
            
            current_transfer_tokens = (x[:, block_start:block_end] == mask_token_id).sum()
            
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
            
            select_index = select_index.to(x.device)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < threshold:
                    transfer_index[0, select_index[0, k]] = False
            
            x[:, block_start:block_end][transfer_index] = x_[transfer_index]
            return x

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        prompt_length = input_ids.shape[1]
        generated_length = 0
        nfe_history = []
        block_history = []
        past_key_values = None
        # Process each block
        while generated_length < gen_length:
            nfe = 0
            block_start = prompt_length + generated_length

            model_output = self(x, attention_mask, tok_idx if tok_idx is not None else None, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            nfe += 1

            # block length prediction
            current_block_length = self._compute_block_length(logits, prompt_length, gen_length, generated_length, block_length, delimiter_threshold=delimiter_threshold)
            block_history.append(current_block_length)
            generated_length += current_block_length
            block_end = block_start + current_block_length

            # mandatory decode first token
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, block_start] = x0[:, block_start]

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = 1

            while True:
                if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                    break

                # sample window
                mask_index = (x[:, block_start:block_end] == mask_token_id)
                
                # prepare attention mask
                if attention_mask != "full":
                    # adjust attention mask
                    current_attention_mask = attention_mask[:, :, :, block_start:]
                else:
                    current_attention_mask = attention_mask

                model_output = self(x[:, block_start:block_end], current_attention_mask, 
                                    tok_idx[:, block_start:block_end] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True, dual_cache=dual_cache, replace_position=replace_position)
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                nfe += 1

                # sample
                if alg == 'confidence_threshold':
                    x = _confidence_threshold_sample(
                        logits=logits,
                        mask_index=mask_index,
                        x=x,
                        block_start=block_start,
                        block_end=block_end,
                        mask_token_id=mask_token_id,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        threshold=threshold
                    )
                else: 
                    raise NotImplementedError(alg)
            
            nfe_history.append(nfe)
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
                block_history=block_history
            )
        else:
            return x