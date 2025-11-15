"""
Unified Ablation Study: Effect of Removing Bad Attempts

This script loads an ICRL checkpoint from any benchmark (Mathador, AIME, or Olympiads)
and studies the effect of progressively removing worst-performing attempts on the 
quality of new inferences.

Usage:
    # With YAML config file (recommended):
    python ablation_study.py ablation_config.yaml
    
    # With CLI arguments:
    python ablation_study.py benchmark_type=mathador \\
                            checkpoint_path=path/to/checkpoint.pkl \\
                            config_path=path/to/config.yaml \\
                            output_dir=path/to/output
    
    # Override YAML config with CLI:
    python ablation_study.py ablation_config.yaml parallelization_degree=5
"""

import os
import sys
import logging
import colorama
from pathlib import Path
from typing import Optional
import anyio
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import numpy as np
import dotenv

# Import ICRL framework
from icrl import ICRLState, Attempt

dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def parse_config():
    """Parse configuration from command line and YAML"""
    # Create base config schema
    base_config = OmegaConf.create({
        'benchmark_type': None,  # Required: 'mathador', 'aime', or 'olympiads'
        'checkpoint_path': None,
        'config_path': None,
        'output_dir': None,
        'parallelization_degree': 10,
        'api_key': None,
        'model_name': None,
        'api_base': None,
        'temperature': None,
        'judge_model_name': None,
        'judge_api_base': None,
        'judge_temperature': None,
    })
    
    # Check if a YAML config file is provided as first argument
    yaml_config = OmegaConf.create()
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        yaml_config = OmegaConf.load(sys.argv[1])
        # Remove the yaml file from argv so it's not parsed again
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    # Parse CLI
    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, yaml_config, cli_conf)
    
    # Validate required arguments
    if config.benchmark_type is None:
        raise ValueError("benchmark_type is required! Usage: benchmark_type=mathador/aime/olympiads")
    if config.benchmark_type not in ['mathador', 'aime', 'olympiads']:
        raise ValueError(f"Invalid benchmark_type: {config.benchmark_type}. Must be mathador, aime, or olympiads")
    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path is required! Usage: checkpoint_path=path/to/checkpoint.pkl")
    if config.output_dir is None:
        raise ValueError("output_dir is required! Usage: output_dir=path/to/output")
    if config.config_path is None:
        raise ValueError("config_path is required! Usage: config_path=path/to/config.yaml")
    
    # Load original config to get model settings
    original_config = OmegaConf.load(config.config_path)
    
    # Only override with non-None values from ablation config
    # This preserves api_base, model settings, etc. from original config
    override_config = OmegaConf.create({
        k: v for k, v in config.items() 
        if v is not None
    })
    
    # Merge with original config (ablation config overrides original for non-None values)
    final_config = OmegaConf.merge(original_config, override_config)
    
    return final_config


def setup_logging(output_dir: str):
    """Setup logging with color and file output"""
    colorama.init()
    
    # Import ColoredFormatter from appropriate benchmark
    # All benchmarks have identical ColoredFormatter, so we can import from any
    from mathador_bench import ColoredFormatter
    
    handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(message)s'))
    handlers.append(console_handler)
    
    # File handler without colors
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "ablation_output.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    handlers.append(file_handler)
    
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    
    # Set log levels for noisy modules
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


# ============================================================================
# Dynamic Imports
# ============================================================================

def get_benchmark_components(benchmark_type: str):
    """
    Dynamically import Task and PromptBuilder classes based on benchmark type.
    
    Returns:
        (Task class, PromptBuilder class, needs_judge flag)
    """
    if benchmark_type == 'mathador':
        from mathador_bench import MathadorTask, MathadorPromptBuilder
        return MathadorTask, MathadorPromptBuilder, False
    elif benchmark_type == 'aime':
        from aime_bench import AIMETask, AIMEPromptBuilder
        return AIMETask, AIMEPromptBuilder, True
    elif benchmark_type == 'olympiads':
        from olympiads_bench import OlympiadsTask, OlympiadsPromptBuilder
        return OlympiadsTask, OlympiadsPromptBuilder, True
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


# ============================================================================
# Ablation Study Main Logic
# ============================================================================

async def run_ablation_study(
    state: ICRLState,
    task,
    prompt_builder,
    llm_call,
    encoder,
    config
):
    """
    Run the ablation study: progressively remove worst attempts and generate new inferences.
    
    For each problem:
        For i in range(1, len(attempts)):
            - Remove the i worst attempts
            - Generate 1 new inference with the remaining attempts
            - Store the new attempt with ablation metadata
    """
    logger.info(f"Starting ablation study on {len(state.problem_histories)} problems")
    
    # Determine max number of iterations based on minimum attempts across problems
    min_attempts = min(len(ph.attempts) for ph in state.problem_histories)
    max_i = min_attempts - 1  # We need at least 1 attempt to show as context
    
    if max_i < 1:
        logger.error(f"Not enough attempts to run ablation study! Min attempts: {min_attempts}")
        return
    
    logger.info(f"Will run ablation for i=1 to i={max_i} (removing 1 to {max_i} worst attempts)")
    
    # Run ablation for each value of i
    for i in range(1, max_i + 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"ABLATION ITERATION: Removing {i} worst attempt(s)")
        logger.info(f"{'='*100}\n")
        
        await run_ablation_iteration(i, state, task, prompt_builder, llm_call, encoder, config)
    
    logger.info("\nAblation study complete!")


async def run_ablation_iteration(
    i: int,
    state: ICRLState,
    task,
    prompt_builder,
    llm_call,
    encoder,
    config
):
    """
    Run one iteration of the ablation study where we remove i worst attempts.
    
    Args:
        i: Number of worst attempts to remove
        state: ICRL state with all problem histories
        task: Task for scoring
        prompt_builder: Prompt builder for formatting
        llm_call: LLM API call function
        encoder: Tokenizer
        config: Configuration
    """
    
    async def ablation_interaction(problem_idx: int):
        """Run ablation for a single problem"""
        problem_history = state.problem_histories[problem_idx]
        problem = problem_history.problem
        
        # Get all attempts (excluding previously generated ablation attempts)
        original_attempts = [
            a for a in problem_history.attempts 
            if 'ablation_i' not in a.extra_fields
        ]
        
        if len(original_attempts) < i + 1:
            logger.debug(f"Problem {problem_idx}: Not enough attempts ({len(original_attempts)}), skipping")
            return
        
        # Sort by reward (ascending) and remove the i worst attempts
        sorted_attempts = sorted(original_attempts, key=lambda x: x.reward)
        filtered_attempts = sorted_attempts[i:]  # Keep the best (len - i) attempts
        
        logger.debug(f"Problem {problem_idx}: Keeping {len(filtered_attempts)}/{len(original_attempts)} attempts")
        
        # Calculate available context length
        available_length = (
            state.config.context_size 
            - state.config.max_completion_tokens 
            - state.config.context_size_safety_margin
        )
        
        # Build prompt using filtered attempts (exploitation mode)
        messages = prompt_builder.build_prompt(
            problem=problem,
            attempts=filtered_attempts,
            is_exploration=False,  # Exploitation mode
            encoder=encoder,
            max_length=available_length,
            max_attempts_in_context=state.config.max_attempts_in_context,
            max_attempt_length=state.config.max_attempt_length
        )
        
        # Get model output
        model_output = await llm_call(messages)
        
        # Score the attempt
        reward = await task.score(model_output, problem)
        
        # Store the attempt with ablation metadata
        attempt = Attempt(
            prompt=messages,
            output=model_output,
            reward=reward,
            round_idx=-2,  # Special marker for ablation attempts
            extra_fields={'ablation_i': i}
        )
        problem_history.attempts.append(attempt)
        
        if problem_idx == 0:
            logger.info(f"Problem 0 (ablation i={i}): Output preview: {model_output[:200]}...")
            logger.info(f"Problem 0 (ablation i={i}): Reward = {reward:.3f}")
    
    # Run in parallel with limited concurrency
    async with anyio.create_task_group() as tg:
        semaphore = anyio.Semaphore(config.parallelization_degree)
        
        async def run_with_semaphore(idx):
            async with semaphore:
                await ablation_interaction(idx)
        
        for idx in range(len(state.problem_histories)):
            tg.start_soon(run_with_semaphore, idx)
    
    # Log statistics for this iteration
    rewards = []
    for ph in state.problem_histories:
        ablation_attempts = [
            a for a in ph.attempts 
            if a.extra_fields.get('ablation_i') == i
        ]
        if ablation_attempts:
            rewards.append(ablation_attempts[0].reward)
    
    if rewards:
        logger.info(
            f"Ablation i={i} rewards - "
            f"25th: {np.percentile(rewards, 25):.3f}, "
            f"50th: {np.percentile(rewards, 50):.3f}, "
            f"75th: {np.percentile(rewards, 75):.3f}, "
            f"Mean: {np.mean(rewards):.3f}"
        )


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution function"""
    # Parse configuration
    config = parse_config()
    
    # Setup logging
    setup_logging(config.output_dir)
    
    benchmark_name = config.benchmark_type.upper()
    logger.info("="*100)
    logger.info(f"ABLATION STUDY: Effect of Removing Bad Attempts ({benchmark_name})")
    logger.info("="*100)
    logger.info(f"Benchmark type: {config.benchmark_type}")
    logger.info(f"Loading checkpoint from: {config.checkpoint_path}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state = ICRLState.load(checkpoint_path)
    logger.info(f"Loaded state with {len(state.problem_histories)} problems")
    
    # Log statistics about loaded state
    all_attempts = [a for ph in state.problem_histories for a in ph.attempts]
    logger.info(f"Total attempts in checkpoint: {len(all_attempts)}")
    logger.info(f"Attempts per problem: min={min(len(ph.attempts) for ph in state.problem_histories)}, "
                f"max={max(len(ph.attempts) for ph in state.problem_histories)}, "
                f"mean={np.mean([len(ph.attempts) for ph in state.problem_histories]):.1f}")
    
    # Get benchmark-specific components
    logger.info(f"\nLoading {config.benchmark_type} benchmark components...")
    TaskClass, PromptBuilderClass, needs_judge = get_benchmark_components(config.benchmark_type)
    
    # Setup LLM clients
    logger.info("Setting up LLM clients...")
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found! Set OPENAI_API_KEY or pass api_key=...")
    
    generation_client = AsyncOpenAI(base_url=config.api_base, api_key=api_key)
    judge_client = None
    
    if needs_judge:
        judge_client = AsyncOpenAI(base_url=config.judge_api_base, api_key=api_key)
    
    # Reconstruct task based on benchmark type
    logger.info("Reconstructing task and prompt builder...")
    if config.benchmark_type == 'mathador':
        task = TaskClass(
            dataset_path=config.dataset_path,
            num_problems=config.num_problems,
            num_shots=config.num_shots
        )
    else:  # aime or olympiads
        task = TaskClass(
            dataset_name=config.dataset_name,
            num_problems=config.num_problems,
            judge_client=judge_client,
            judge_model_name=config.judge_model_name,
            judge_temperature=config.judge_temperature
        )
    
    # Force load problems to ensure same problems as checkpoint
    _ = task.get_problems()
    
    prompt_builder = PromptBuilderClass()
    
    # Create LLM call wrapper for generation
    async def llm_call(messages: list[dict]) -> str:
        """Wrapper for LLM API calls"""
        response = await generation_client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_completion_tokens,
        )
        return response.choices[0].message.content
    
    # Setup encoder
    logger.info("Loading tokenizer...")
    try:
        encoder = AutoTokenizer.from_pretrained(config.model_encoder)
    except Exception as e:
        logger.warning(f"Could not load tokenizer for {config.model_encoder}, using GPT-2 as fallback: {e}")
        encoder = AutoTokenizer.from_pretrained("gpt2")
    
    # Run ablation study
    await run_ablation_study(state, task, prompt_builder, llm_call, encoder, config)
    
    # Save results
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "checkpoint_ablation.pkl"
    state.save(result_path)
    logger.info(f"\nSaved ablation results to: {result_path}")
    
    # Report final statistics
    logger.info("\n" + "="*100)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*100)
    
    # For each ablation value, compute statistics
    max_i = max(
        a.extra_fields.get('ablation_i', 0) 
        for ph in state.problem_histories 
        for a in ph.attempts
    )
    
    logger.info("\nRewards by number of removed attempts:")
    logger.info(f"{'Removed':<10} {'Mean':<10} {'25th':<10} {'50th':<10} {'75th':<10} {'Count':<10}")
    logger.info("-" * 60)
    
    for i in range(1, max_i + 1):
        rewards = [
            a.reward 
            for ph in state.problem_histories 
            for a in ph.attempts 
            if a.extra_fields.get('ablation_i') == i
        ]
        if rewards:
            logger.info(
                f"{i:<10} {np.mean(rewards):<10.3f} {np.percentile(rewards, 25):<10.3f} "
                f"{np.percentile(rewards, 50):<10.3f} {np.percentile(rewards, 75):<10.3f} "
                f"{len(rewards):<10}"
            )
    
    # Compare with original attempts
    original_rewards = [
        a.reward 
        for ph in state.problem_histories 
        for a in ph.attempts 
        if 'ablation_i' not in a.extra_fields
    ]
    
    logger.info("\nOriginal attempts (for comparison):")
    logger.info(
        f"{'Original':<10} {np.mean(original_rewards):<10.3f} "
        f"{np.percentile(original_rewards, 25):<10.3f} "
        f"{np.percentile(original_rewards, 50):<10.3f} "
        f"{np.percentile(original_rewards, 75):<10.3f} "
        f"{len(original_rewards):<10}"
    )
    
    logger.info(f"\n{benchmark_name} ablation study complete!")


if __name__ == "__main__":
    anyio.run(main)

