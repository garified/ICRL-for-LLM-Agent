"""
Shared boilerplate for ICRL benchmark runners.

Provides: ColoredFormatter, setup_logging, parse_config, make_llm_call,
load_encoder, make_icrl_config, report_results, run_benchmark.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime

import colorama
import dotenv
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from icrl import ICRL, ICRLConfig

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Logging
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{colorama.Fore.GREEN}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{colorama.Fore.YELLOW}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{colorama.Fore.RED}{record.msg}{colorama.Fore.RESET}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{colorama.Fore.RED}{colorama.Style.BRIGHT}{record.msg}{colorama.Style.RESET_ALL}"
        return super().format(record)


def setup_logging(config):
    """Setup logging with color console and optional file output."""
    colorama.init()

    handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(message)s'))
    handlers.append(console_handler)

    # File handler without colors
    if not config.debug_run:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_file = output_path / "output.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)

    # Set log levels for noisy modules
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


# ============================================================================
# Config helpers
# ============================================================================

def parse_config(config_cls):
    """
    Parse configuration from CLI and optional YAML file.

    Args:
        config_cls: dataclass type used as the default config schema
    """
    default_config = OmegaConf.structured(config_cls)

    yaml_config = OmegaConf.create()
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        yaml_config = OmegaConf.load(sys.argv[1])
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(default_config, yaml_config, cli_conf)

    # Debug mode adjustments
    if config.debug_run:
        config.num_problems = 2
        config.num_rounds = 3
        config.num_initial_attempts = 1
        config.max_completion_tokens = 200
        config.parallelization_degree = 2
        logger.info("*" * 100)
        logger.info("DEBUG MODE")
        logger.info("*" * 100)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") + "_" + str(uuid.uuid4())[:3]
    if config.postfix:
        timestamp = timestamp + "_" + config.postfix

    output_path = Path(config.base_output_dir) / timestamp
    config.output_dir = str(output_path)

    if not config.debug_run:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "config.yaml", "w") as f:
            OmegaConf.save(config, f)

    return config


# ============================================================================
# LLM / tokenizer helpers
# ============================================================================

def make_llm_call(client, model_name, temperature, max_tokens):
    """Return an async closure that calls the OpenAI-compatible API."""

    async def llm_call(messages: list[dict]) -> str:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    return llm_call


def load_encoder(model_encoder: str):
    """Load a tokenizer, falling back to GPT-2 on failure."""
    try:
        return AutoTokenizer.from_pretrained(model_encoder)
    except Exception as e:
        logger.warning(f"Could not load tokenizer for {model_encoder}, using GPT-2 as fallback: {e}")
        return AutoTokenizer.from_pretrained("gpt2")


# ============================================================================
# ICRL wiring
# ============================================================================

def make_icrl_config(config) -> ICRLConfig:
    """Extract an ICRLConfig from any benchmark config dataclass."""
    return ICRLConfig(
        num_initial_attempts=config.num_initial_attempts,
        num_rounds=config.num_rounds,
        max_completion_tokens=config.max_completion_tokens,
        context_size=config.context_size,
        parallelization_degree=config.parallelization_degree,
        max_attempts_in_context=config.max_attempts_in_context,
        max_attempt_length=config.max_attempt_length,
        checkpoint_dir=config.output_dir if not config.debug_run else None,
    )


def report_results(problem_histories, max_score_display: int = 100):
    """Log final statistics after a benchmark run."""
    logger.info("\n" + "=" * 100)
    logger.info("FINAL RESULTS")
    logger.info("=" * 100)

    all_best_rewards = []
    for i, ph in enumerate(problem_histories):
        if ph.attempts:
            best_reward = max(attempt.reward for attempt in ph.attempts)
            all_best_rewards.append(best_reward)
            if i < 5:
                logger.info(
                    f"Problem {i}: Best reward = {best_reward:.3f} "
                    f"(score: {int(best_reward * max_score_display)}/{max_score_display})"
                )

    if all_best_rewards:
        logger.info(f"\nOverall statistics:")
        logger.info(f"  Mean best reward: {sum(all_best_rewards) / len(all_best_rewards):.3f}")
        logger.info(f"  25th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//4]:.3f}")
        logger.info(f"  50th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//2]:.3f}")
        logger.info(f"  75th percentile: {sorted(all_best_rewards)[3*len(all_best_rewards)//4]:.3f}")
        logger.info(
            f"  Problems with perfect score ({max_score_display}/{max_score_display}): "
            f"{sum(1 for r in all_best_rewards if r >= 0.99)}"
        )


async def run_benchmark(config, task, prompt_builder, llm_call, encoder, name: str, max_score_display: int = 100):
    """
    Universal main loop shared by all math benchmarks.

    Args:
        config: benchmark-specific config (must have the standard ICRL fields)
        task: Task instance
        prompt_builder: PromptBuilder instance
        llm_call: async callable for LLM inference
        encoder: tokenizer
        name: human-readable benchmark name for log messages
        max_score_display: denominator when printing scores (100 for AIME/Olympiads, 18 for Mathador)
    """
    logger.info(f"Starting {name} ICRL benchmark")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    icrl_config = make_icrl_config(config)

    icrl = ICRL(
        task=task,
        prompt_builder=prompt_builder,
        llm_call=llm_call,
        encoder=encoder,
        config=icrl_config,
    )

    final_state = await icrl.solve()

    report_results(final_state.problem_histories, max_score_display)
    logger.info(f"\n{name} ICRL benchmark complete!")
