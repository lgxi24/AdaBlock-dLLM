# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Base-7B"
length=512
block_length=32
threshold=0.9
delimiter_threshold=0.5
steps=$((length / block_length))

# ============== Math Tasks (gsm8k, minerva_math) ==============
task=gsm8k
num_fewshot=5

# adablock_no_cache
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=${delimiter_threshold},show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code

# adablock_dual_cache
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=${delimiter_threshold},use_cache=true,dual_cache=true,show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code


# ============== Coding Tasks (humaneval, mbpp) ==============
task=humaneval
num_fewshot=0

# adablock_no_cache
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=${delimiter_threshold},show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path eval_results_adablock/${task}/no_cache --log_samples

# adablock_dual_cache
accelerate launch eval_dream_adablock.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},delimiter_threshold=${delimiter_threshold},use_cache=true,dual_cache=true,show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path eval_results_adablock/${task}/dual_cache --log_samples
