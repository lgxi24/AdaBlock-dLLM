# LLaDA Model Evaluation Guide

This document provides instructions for evaluating the LLaDA model using baseline (LLaDA and Fast-dLLM) and AdaBlock-dLLM.

## Environment Setup

Before running any evaluation, set the following environment variables:
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```



## Baseline Evaluation (`eval_llada_baseline.py`)

### Common Parameters
```bash
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"  # or "GSAI-ML/LLaDA-1.5-8B-Instruct"
GEN_LENGTH=512
BLOCK_LENGTH=32
THRESHOLD=0.9
```

### Methods

#### 1. Top-K Decoding
- `steps = gen_length` (one token per step)

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},show_speed=True \
--output_path eval_results/humaneval/baseline --log_samples
```

#### 2. Threshold-based Parallel Decoding (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9` for parallel decoding

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True \
--output_path eval_results/humaneval/thres_no_cache --log_samples
```

#### 3. Threshold-based Parallel Decoding + Prefix Cache (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9`, `use_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results/humaneval/thres_prefix_cache --log_samples
```

#### 4. Threshold-based Parallel Decoding + Dual Cache (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9`, `use_cache=True`, `dual_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results/humaneval/thres_dual_cache --log_samples
```

## AdaBlock Evaluation (`eval_llada_adablock.py`)

AdaBlock uses adaptive block length based on delimiter confidence.

### Common Parameters
```bash
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"  # or "GSAI-ML/LLaDA-1.5-8B-Instruct"
GEN_LENGTH=512
INIT_BLOCK_LENGTH=32
THRESHOLD=0.9
```

### AdaBlock-specific Parameters
```bash
DELIMITER_THRESHOLD=0.3
DELIMITER_IDS="198"  # 198=newline; for multiple delimiters: "198,11,13"
```

### Methods

#### 1. AdaBlock + Fast-dLLM (No Cache)
- `delimiter_threshold=0.3` for adaptive block length

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True \
--output_path eval_results_adablock/humaneval/no_cache --log_samples
```

#### 2. AdaBlock + Fast-dLLM (Prefix Cache)
- `delimiter_threshold=0.3`
- `use_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results_adablock/humaneval/prefix_cache --log_samples
```

#### 3. AdaBlock + Fast-dLLM (Dual Cache)
- `delimiter_threshold=0.3`
- `use_cache=True`, `dual_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results_adablock/humaneval/dual_cache --log_samples
```

## Parameter Reference

### Common Parameters
| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to the LLaDA model |
| `gen_length` | Total generation length |
| `steps` | Number of denoising steps, used if denoising step is fixed |
| `block_length` | Block size for semi-AR Decoding |
| `threshold` | Confidence threshold for token transfer |
| `show_speed` | Display speed metrics |

### Cache Parameters
| Parameter | Description |
|-----------|-------------|
| `use_cache` | Enable block-level KV cache (defaults to prefix cache) |
| `dual_cache` | Use dual cache instead of prefix cache |

### AdaBlock-specific Parameters
| Parameter | Description |
|-----------|-------------|
| `delimiter_ids` | Token IDs for delimiters (e.g., 198=newline, 11=comma, 13=period; refer to [LLaDA tokenizer](https://huggingface.co/GSAI-ML/LLaDA-8B-Base/blob/main/tokenizer.json)) |
| `delimiter_threshold` | Confidence threshold for adaptive block length |

## Post-processing (HumanEval/MBPP)

For code generation tasks, post-processing is required:
```bash
# For HumanEval
python postprocess_humaneval.py {samples_xxx.jsonl}

# For MBPP
python postprocess_mbpp.py {samples_xxx.jsonl}
```

## Notes

1. Run scripts from the `llada/` directory
2. Batch evaluation is currently not supported (`batch_size` must be 1)
3. For HumanEval/MBPP, samples are logged with `--log_samples` for post-processing
4. If GSM8K dataset fails to load (refer to this [issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3528)), update `dataset_path` in `lm_eval/tasks/gsm8k/*.yaml`:
   ```bash
   # Find your lm_eval installation path
   python -c "import lm_eval; print(lm_eval.__path__[0])"
   
   # Update all gsm8k yaml files (replace <path> with the output above)
   sed -i 's/dataset_path: gsm8k/dataset_path: openai\/gsm8k/g' <path>/tasks/gsm8k/*.yaml
   ```
