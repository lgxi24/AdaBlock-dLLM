# Dream Model Evaluation Guide

This document provides instructions for evaluating the Dream model using baseline (Dream and Fast-dLLM) and AdaBlock-dLLM.

## Environment Setup

Before running any evaluation, set the following environment variables:
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```



## Baseline Evaluation (`eval_dream.py`)

### Common Parameters
```bash
model="Dream-org/Dream-v0-Base-7B"
length=512
block_length=32
steps=$((length / block_length))
```

### Methods

#### 1. Top-1 Decoding (Baseline)
- `threshold=1.0` for top-1 sampling

```bash
# GSM8K
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=1.0,show_speed=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --confirm_run_unsafe_code

# HumanEval
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=1.0,show_speed=True,escape_until=true \
    --tasks humaneval --num_fewshot 0 --batch_size 1 --confirm_run_unsafe_code \
    --output_path eval_results/humaneval/baseline --log_samples
```

#### 2. Threshold-based Parallel Decoding (Fast-dLLM)
- `threshold=0.9` for parallel decoding

```bash
# GSM8K
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --confirm_run_unsafe_code

# HumanEval
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,escape_until=true \
    --tasks humaneval --num_fewshot 0 --batch_size 1 --confirm_run_unsafe_code \
    --output_path eval_results/humaneval/thres_no_cache --log_samples
```

#### 3. Threshold-based Parallel Decoding + Dual Cache (Fast-dLLM)
- `threshold=0.9`, `use_cache=true`, `dual_cache=true`

```bash
# GSM8K
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --confirm_run_unsafe_code

# HumanEval
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,escape_until=true \
    --tasks humaneval --num_fewshot 0 --batch_size 1 --confirm_run_unsafe_code \
    --output_path eval_results/humaneval/thres_dual_cache --log_samples
```

## AdaBlock Evaluation (`eval_dream_adablock.py`)

AdaBlock uses adaptive block length based on delimiter confidence.

### Common Parameters
```bash
model="Dream-org/Dream-v0-Base-7B"
length=512
block_length=32
threshold=0.9
steps=$((length / block_length))
```

### AdaBlock-specific Parameters
```bash
delimiter_threshold=0.5
# delimiter_ids are hardcoded in the model (newline tokens)
```

### Methods

#### 1. AdaBlock + Fast-dLLM (No Cache)
- `delimiter_threshold=0.5` for adaptive block length

```bash
# GSM8K
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=0.5,show_speed=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --confirm_run_unsafe_code

# HumanEval
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=0.5,show_speed=True,escape_until=true \
    --tasks humaneval --num_fewshot 0 --batch_size 1 --confirm_run_unsafe_code \
    --output_path eval_results_adablock/humaneval/no_cache --log_samples
```

#### 2. AdaBlock + Fast-dLLM (Dual Cache)
- `delimiter_threshold=0.5`
- `use_cache=true`, `dual_cache=true`

```bash
# GSM8K
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=0.5,use_cache=true,dual_cache=true,show_speed=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --confirm_run_unsafe_code

# HumanEval
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=0.5,use_cache=true,dual_cache=true,show_speed=True,escape_until=true \
    --tasks humaneval --num_fewshot 0 --batch_size 1 --confirm_run_unsafe_code \
    --output_path eval_results_adablock/humaneval/dual_cache --log_samples
```

## Parameter Reference

### Common Parameters
| Parameter | Description |
|-----------|-------------|
| `pretrained` | Path to the Dream model |
| `max_new_tokens` | Total generation length |
| `diffusion_steps` | Number of denoising steps |
| `block_length` | Block size for semi-AR Decoding |
| `threshold` | Confidence threshold for token transfer |
| `add_bos_token` | Add BOS token to input (required for Dream) |
| `alg` | Generation algorithm: `entropy` (vanilla Dream) or `confidence_threshold` (Fast-dLLM) |
| `show_speed` | Display speed metrics |

### Cache Parameters
| Parameter | Description |
|-----------|-------------|
| `use_cache` | Enable block-level KV cache |
| `dual_cache` | Use dual cache (required with `use_cache=true`) |

### AdaBlock-specific Parameters
| Parameter | Description |
|-----------|-------------|
| `delimiter_threshold` | Confidence threshold for adaptive block length |

### Code Generation Parameters
| Parameter | Description |
|-----------|-------------|
| `escape_until` | Enable escape until for code generation (required for HumanEval/MBPP) |

## Post-processing (HumanEval/MBPP)

For code generation tasks, post-processing is required:
```bash
# For HumanEval
python postprocess_humaneval.py {samples_xxx.jsonl}

# For MBPP
python postprocess_mbpp.py {samples_xxx.jsonl}
```

## Notes

1. Run scripts from the `dream/` directory
2. Batch evaluation is currently not supported (`batch_size` must be 1)
3. For HumanEval/MBPP, samples are logged with `--log_samples` for post-processing
4. Vanilla Dream uses `alg=entropy`, here we use `alg=confidence_threshold` with `threshold=1.0` for fair comparison
5. If GSM8K dataset fails to load (refer to this [issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3528)), update `dataset_path` in `lm_eval/tasks/gsm8k/*.yaml`:
   ```bash
   # Find your lm_eval installation path
   python -c "import lm_eval; print(lm_eval.__path__[0])"
   
   # Update all gsm8k yaml files (replace <path> with the output above)
   sed -i 's/dataset_path: gsm8k/dataset_path: openai\/gsm8k/g' <path>/tasks/gsm8k/*.yaml
   ```
