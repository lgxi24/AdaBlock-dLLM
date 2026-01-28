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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA
# Modified by Guanxi Lu, Imperial College London

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def compute_block_length(
    logits,                
    predicted_tokens,      
    prompt,                
    gen_length,       
    generated_length,
    default_block_length,
    delimiter_ids=[198],  # default: newline token (11=comma, 13=period, 198=newline)
    delimiter_threshold=float('inf')
):
    """
    Compute adaptive block length based on delimiter confidence.
    Returns the position of the highest-confidence delimiter if above threshold,
    otherwise returns default_block_length.
    """
    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    
    # Create sampling window (25% of gen_length, capped by remaining)
    window_size = min(int(0.25 * gen_length), remaining_length)
    window_tokens = predicted_tokens[0, block_start:block_start + window_size]
    
    # Create mask for delimiter tokens
    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= (window_tokens == token_id)

    # Fallback to default block length if no delimiter is found
    if not torch.any(delimiter_mask):
        return min(default_block_length, remaining_length)

    # Get positions of delimiters in the sequence
    delimiter_pos = block_start + torch.nonzero(delimiter_mask).squeeze(-1)
    
    # Compute confidence for each delimiter
    delimiter_logits = logits[0, delimiter_pos, predicted_tokens[0, delimiter_pos]]
    log_sum_exp = torch.logsumexp(logits[0, delimiter_pos, :], dim=-1)
    delimiter_confidences = torch.exp(delimiter_logits - log_sum_exp)

    # Find the delimiter with highest confidence
    max_confidence, best_idx = torch.max(delimiter_confidences, dim=0)
    max_confidence = max_confidence.item()
    best_delimiter_pos = delimiter_pos[best_idx].item()

    if max_confidence >= delimiter_threshold:
        block_length = best_delimiter_pos - block_start + 1
    else:
        block_length = min(default_block_length, remaining_length)
    return block_length

@ torch.no_grad()
def generate_adablock(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf')):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []
    while generated_length < gen_length: 
        nfe = 0

        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        
        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x, None, threshold)
            x[transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history

@torch.no_grad()
def generate_adablock_prefix_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf')):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []

    while generated_length < gen_length: 
        nfe = 0

        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        # truncate cache to prefix only (before current block)
        prefix_cache = []
        for i in range(len(full_cache)):
            prefix_cache.append(())
            for j in range(len(full_cache[i])):
                prefix_cache[i] += (full_cache[i][j][:, :, :block_start],)

        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:], None, threshold)
            x[:, block_start:][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_adablock_dual_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf')): 
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []
    
    while generated_length < gen_length: 
        nfe = 0

        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        
        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:block_end] == mask_id)
            block_output = model(x[:, block_start:block_end], past_key_values=full_cache, use_cache=True, replace_position=replace_position)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:block_end], None, threshold)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history

def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    x0 = predicted_tokens  # (B, L)

    # Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out_ids, nfe_history, block_history = generate_adablock(model, input_ids, steps=32, gen_length=256, init_block_length=16, temperature=0., remasking='low_confidence', threshold=0.9, delimiter_ids=[198], delimiter_threshold=0.3)
    
    print(tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print()
    print(f"NFE for each block: {nfe_history}")
    print(f"Total NFE: {sum(nfe_history)}")
    print(f"Average NFE per block: {sum(nfe_history) / len(nfe_history)}")
    print()
    print(f"Block length for each block: {block_history}")
    print(f"Number of blocks: {len(block_history)}")
    print(f"Average block length: {sum(block_history) / len(block_history)}")

if __name__ == '__main__':
    main()
