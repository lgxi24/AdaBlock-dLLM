#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ============== Common Settings ==============
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
GEN_LENGTH=512
INIT_BLOCK_LENGTH=32
THRESHOLD=0.9
DELIMITER_THRESHOLD=0.3
DELIMITER_IDS="198"  # for multiple delimiters, use a list like [198, 11, 13]

# ============== Math Tasks (gsm8k, minerva_math) ==============
TASK="gsm8k"
NUM_FEWSHOT=5
STEPS=$((GEN_LENGTH / INIT_BLOCK_LENGTH))

# adablock_no_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True

# adablock_prefix_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True

# adablock_dual_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True


# ============== Coding Tasks (humaneval, mbpp) ==============
TASK="humaneval"
NUM_FEWSHOT=0
STEPS=$((GEN_LENGTH / INIT_BLOCK_LENGTH))

# adablock_no_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True \
--output_path eval_results_adablock/${TASK}/no_cache --log_samples

# adablock_prefix_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results_adablock/${TASK}/prefix_cache --log_samples

# adablock_dual_cache
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks ${TASK} --num_fewshot ${NUM_FEWSHOT} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results_adablock/${TASK}/dual_cache --log_samples
