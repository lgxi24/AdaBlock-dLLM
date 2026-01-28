# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Base-7B"
length=512
block_length=32
steps=$((length / block_length))

# ============== Math Tasks (gsm8k, minerva_math) ==============
task=gsm8k
num_fewshot=5

# baseline (threshold=1.0, top-1 sampling)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=1.0,show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code

# thres_no_cache (threshold=0.9)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code

# thres_dual_cache (threshold=0.9)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code


# ============== Coding Tasks (humaneval, mbpp) ==============
task=humaneval
num_fewshot=0

# baseline (threshold=1.0, top-1 sampling)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=1.0,show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path eval_results/${task}/baseline --log_samples

# thres_no_cache (threshold=0.9)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path eval_results/${task}/thres_no_cache --log_samples

# thres_dual_cache (threshold=0.9)
accelerate launch eval_dream.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path eval_results/${task}/thres_dual_cache --log_samples
