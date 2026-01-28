#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ============== Common Settings ==============
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
GEN_LENGTH=512
BLOCK_LENGTH=32
THRESHOLD=0.9

# ============== Math Tasks (gsm8k, minerva_math) ==============
TASK="gsm8k"
NUM_FEWSHOT=5

# baseline (steps = gen_length)
STEPS=${GEN_LENGTH}
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True

# thres_no_cache (steps = gen_length / block_length)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True

# thres_prefix_cache (steps = gen_length / block_length, with prefix cache)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True

# thres_dual_cache (steps = gen_length / block_length, with dual cache)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True


# ============== Coding Tasks (humaneval, mbpp) ==============
TASK="humaneval"
NUM_FEWSHOT=0

# baseline (steps = gen_length)
STEPS=${GEN_LENGTH}
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True \
--output_path eval_results/${TASK}/baseline --log_samples

# thres_no_cache (steps = gen_length / block_length)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True \
--output_path eval_results/${TASK}/thres_no_cache --log_samples

# thres_prefix_cache (steps = gen_length / block_length, with prefix cache)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results/${TASK}/thres_prefix_cache --log_samples

# thres_dual_cache (steps = gen_length / block_length, with dual cache)
STEPS=$((GEN_LENGTH / BLOCK_LENGTH))
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results/${TASK}/thres_dual_cache --log_samples
