# vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 --served-model-name "

# python3 llm_game24_api_refactor.py

# python3 run_sciworld.py icrl_mode=ICRL num_envs=10 postfix="temp"
# python3 run_sciworld.py icrl_mode=NO_REWARDS num_envs=10 postfix="no_rewards"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=10 postfix="3_attempts" positive_only=true
# python3 run_sciworld.py icrl_mode=RANDOM_SAMPLING num_envs=10 postfix="random_sampling"
# python3 run_sciworld.py icrl_mode=REFLEXION num_envs=29 postfix="reflexion_29"


# final runs
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 postfix="29_4.1"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 zero_out_rewards=true postfix="29_4.1_zero_out_rewards"
python3 run_sciworld.py icrl_mode=ICRL num_envs=29 max_attempts_in_context=3 postfix="29_4.1_3_icl"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 explore_only=true postfix="29_4.1_explore_only"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 exploit_only=true postfix="29_4.1_exploit_only"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 explore_and_exploit=true postfix="29_4.1_explore_and_exploit"
# python3 run_sciworld.py icrl_mode=ICRL num_envs=29 neutral_prompt=true postfix="29_4.1_neutral_prompt"

# python3 run_sciworld.py icrl_mode=RANDOM_SAMPLING num_envs=29 postfix="29_4.1_random_sampling" max_env_steps=200
# python3 run_sciworld.py icrl_mode=REFLEXION num_envs=29 postfix="29_4.1_reflexion_obsfix"
# python3 run_sciworld.py icrl_mode=REACT num_envs=29 postfix="29_4.1_react"
# python3 run_sciworld.py icrl_mode=SELFREFINE num_envs=29 postfix="29_4.1_selfrefine"
# python3 run_sciworld.py icrl_mode=COT num_envs=29 postfix="29_4.1_cot"

# math

vllm serve Qwen/Qwen3-32B --dtype auto --api-key hi -tp 4 --gpu-memory-utilization 0.65 --port 11435
vllm serve microsoft/phi-4 --dtype auto --api-key hi -tp 4 --gpu-memory-utilization 0.65 --port 11435
vllm serve virtuoussy/Qwen2.5-7B-Instruct-RLVR --dtype auto --api-key hi -tp 4 --gpu-memory-utilization 0.3 --port 11436 --max_model_len 2048

vllm serve virtuoussy/Qwen2.5-7B-Instruct-RLVR --dtype auto --api-key hi -tp 2 --gpu-memory-utilization 0.95 --port 11436 --max_model_len 4096


# qwen3.32b no reasoning
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt disable_reasoning=true
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime disable_reasoning=true
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_selfrefine disable_reasoning=true icrl_mode=SELFREFINE
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_selfrefine disable_reasoning=true icrl_mode=SELFREFINE
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_reflexion disable_reasoning=true icrl_mode=REFLEXION
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_reflexion disable_reasoning=true icrl_mode=REFLEXION

python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_ee disable_reasoning=true max_attempt_length=4096 explore_and_exploit=true

# qwen3.32b reasoning
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_reason vllm_address=http://localhost:11435/v1 
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_reason vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_reason_selfrefine icrl_mode=SELFREFINE vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_reason_selfrefine icrl_mode=SELFREFINE vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_reason_reflexion icrl_mode=REFLEXION vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_reason_reflexion icrl_mode=REFLEXION vllm_address=http://localhost:11435/v1

# phi4
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_phi4 vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/aime_2025 postfix=aime_phi4 vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_phi4_selfrefine icrl_mode=SELFREFINE vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/aime_2025 postfix=aime_phi4_selfrefine icrl_mode=SELFREFINE vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_phi4_reflexion icrl_mode=REFLEXION vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1
python3 math_bench.py model_name=microsoft/phi-4 dataset_name=MathArena/aime_2025 postfix=aime_phi4_reflexion icrl_mode=REFLEXION vllm_context_size=16384 max_completion_tokens=8192 vllm_address=http://localhost:11435/v1

# llama maverick
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_maverick
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/aime_2025 postfix=aime_maverick
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_maverick_selfrefine icrl_mode=SELFREFINE
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/aime_2025 postfix=aime_maverick_selfrefine icrl_mode=SELFREFINE
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/hmmt_feb_2025 postfix=hmmt_maverick_reflexion icrl_mode=REFLEXION
python3 math_bench.py model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct dataset_name=MathArena/aime_2025 postfix=aime_maverick_reflexion icrl_mode=REFLEXION

# ablations
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_prompt0 disable_reasoning=true alternate_prompt=0 vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_prompt1 disable_reasoning=true alternate_prompt=1 vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_prompt2 disable_reasoning=true alternate_prompt=2 vllm_address=http://localhost:11435/v1
python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_prompt3 disable_reasoning=true alternate_prompt=3 vllm_address=http://localhost:11435/v1

python3 math_bench.py dataset_name=MathArena/aime_2025 postfix=aime_random_reward disable_reasoning=true random_rewards=true vllm_address=http://localhost:11435/v1

# mathador

vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen3-next 

vllm serve google/gemma-3-27b-it     --data-parallel-size 8   --served-model-name qwen3-next

python3 run_mathador.py benchmarks/mathador/config.yaml

python -m analysis.ablation analysis/ablation_mathador.yaml

################################################################################

# mathador via openrouter (grok-4.1-fast), 100 problems, 10 rounds
OPENAI_API_KEY=$(grep OPENROUTER_API_KEY .env | cut -d= -f2) python3 run_mathador.py benchmarks/mathador/config.yaml num_problems=100 num_rounds=10 num_initial_attempts=2 max_completion_tokens=4096 context_size=131072 parallelization_degree=10 model_name=x-ai/grok-4.1-fast model_encoder=gpt2 api_base=https://openrouter.ai/api/v1 temperature=1.0 novelty_reward=false postfix=grok_baseline
OPENAI_API_KEY=$(grep OPENROUTER_API_KEY .env | cut -d= -f2) python3 run_mathador.py benchmarks/mathador/config.yaml num_problems=100 num_rounds=10 num_initial_attempts=2 max_completion_tokens=4096 context_size=131072 parallelization_degree=10 model_name=x-ai/grok-4.1-fast model_encoder=gpt2 api_base=https://openrouter.ai/api/v1 temperature=1.0 novelty_reward=true novelty_weight=0.3 novelty_model=Qwen/Qwen3-0.6B novelty_device=cuda postfix=grok_novelty

# mathador via openrouter (qwen3-235b), 100 problems, 10 rounds
OPENAI_API_KEY=$(grep OPENROUTER_API_KEY .env | cut -d= -f2) python3 run_mathador.py benchmarks/mathador/config.yaml num_problems=100 num_rounds=10 num_initial_attempts=2 max_completion_tokens=4096 context_size=131072 parallelization_degree=10 model_name=qwen/qwen3-235b-a22b-2507 model_encoder=gpt2 api_base=https://openrouter.ai/api/v1 temperature=1.0 novelty_reward=false postfix=qwen3_baseline
OPENAI_API_KEY=$(grep OPENROUTER_API_KEY .env | cut -d= -f2) python3 run_mathador.py benchmarks/mathador/config.yaml num_problems=100 num_rounds=10 num_initial_attempts=2 max_completion_tokens=4096 context_size=131072 parallelization_degree=10 model_name=qwen/qwen3-235b-a22b-2507 model_encoder=gpt2 api_base=https://openrouter.ai/api/v1 temperature=1.0 novelty_reward=true novelty_weight=0.3 novelty_model=Qwen/Qwen3-0.6B novelty_device=cuda postfix=qwen3_novelty