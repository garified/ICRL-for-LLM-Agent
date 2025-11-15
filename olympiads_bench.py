"""
Mathematical Olympiads Benchmark Implementation using ICRL Framework
Adapted from AIME benchmark for AI-MO/olympiads-ref dataset
"""

import os
import re
import sys
import json
import logging
import colorama
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import anyio
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_dataset
import dotenv

# Import ICRL framework
from icrl import (
    Task, Problem, Attempt, PromptBuilder, 
    ICRL, ICRLConfig
)

dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OlympiadsConfig:
    """Configuration for Mathematical Olympiads benchmark"""
    # Dataset
    dataset_name: str = "AI-MO/olympiads-ref"
    num_problems: int = -1  # -1 means all problems
    
    # ICRL parameters
    num_initial_attempts: int = 2
    num_rounds: int = 50
    max_completion_tokens: int = 4096
    context_size: int = 262144
    parallelization_degree: int = 100
    max_attempts_in_context: Optional[int] = None
    max_attempt_length: int = 4096  # Olympiad solutions can be long
    
    # Model configuration
    model_name: str = "qwen3-next"
    model_encoder: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    api_base: str = "http://0.0.0.0:8000/v1"
    api_key: Optional[str] = None
    temperature: float = 1.0
    
    # Judge configuration (uses same model)
    judge_model_name: str = "qwen3-next"
    judge_api_base: str = "http://0.0.0.0:8000/v1"
    judge_temperature: float = 0.0  # Use 0 for more consistent scoring
    
    # Output
    base_output_dir: str = "ICL/olympiads/"
    postfix: str = ""
    output_dir: str = ""  # Will be set automatically with timestamp
    debug_run: bool = False


# ============================================================================
# Olympiads Task Implementation
# ============================================================================

class OlympiadsTask(Task):
    """Task implementation for Mathematical Olympiads benchmark"""
    
    def __init__(self, dataset_name: str, num_problems: int = -1, judge_client: AsyncOpenAI = None, judge_model_name: str = "gpt-4", judge_temperature: float = 0.0):
        """
        Initialize Olympiads task.
        
        Args:
            dataset_name: HuggingFace dataset name
            num_problems: Number of problems to use (-1 for all)
            judge_client: AsyncOpenAI client for judge scoring
            judge_model_name: Model name for judge
            judge_temperature: Temperature for judge scoring
        """
        self.dataset_name = dataset_name
        self.num_problems = num_problems
        self.judge_client = judge_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self._problems: Optional[List[Problem]] = None
    
    def get_problems(self) -> List[Problem]:
        """Load and return all problems from the dataset"""
        if self._problems is not None:
            return self._problems
        
        # Load dataset from HuggingFace
        dataset = load_dataset(self.dataset_name, split='train')
        
        # Limit problems if specified
        if self.num_problems != -1:
            dataset = dataset.select(range(min(self.num_problems, len(dataset))))
        
        # Create problem instances
        problems = []
        for data in dataset:
            problem = Problem(
                question=data['problem'],
                reference_answer=None,  # No separate answer field in olympiads-ref
                metadata={
                    'exam': data.get('exam', 'Unknown'),
                    'year': data.get('year', ''),
                    'tier': data.get('tier', ''),
                    'problem_label': data.get('problem_label', ''),
                    'problem_type': data.get('problem_type', ''),
                    'solution': data['solution'],
                }
            )
            problems.append(problem)
        
        self._problems = problems
        logger.info(f"Loaded {len(problems)} problems from {self.dataset_name}")
        return problems
    
    async def score(self, solution: str, problem: Problem) -> float:
        """
        Score a solution using an LLM judge.
        
        Args:
            solution: The model's generated solution
            problem: The problem being solved
            
        Returns:
            Normalized reward between 0 and 1
        """
        reference_solution = problem.metadata['solution']
        exam_info = f"{problem.metadata.get('exam', '')} {problem.metadata.get('year', '')}".strip()
        
        # Create judge prompt
        judge_prompt = f"""You are an expert mathematics judge tasked with evaluating solutions to Mathematical Olympiad problems.

Problem ({exam_info}):
{problem.question}

Reference Solution:
{reference_solution}

Generated Solution:
{solution}

Instructions:
- Evaluate the generated solution based on the correctness of reasoning, approach, and mathematical validity.
- Give partial credit for partially correct solutions or valid approaches with minor errors.
- Give full marks (100) ONLY if the solution is correct AND the reasoning is sound.
- Compare the final answer/conclusion in the generated solution with the reference solution.
- If the final answer/conclusion is incorrect, the maximum score should be less than 100, even if the approach has merit.

Provide a score from 0 to 100, where:
- 0 = Completely incorrect or no valid mathematical content
- 1-30 = Some relevant mathematical content but fundamentally flawed
- 31-60 = Partially correct approach with significant errors
- 61-90 = Mostly correct with minor errors or incomplete reasoning
- 91-99 = Nearly perfect but final conclusion is incorrect or minor issue
- 100 = Perfect solution with correct conclusion and sound reasoning

Respond with your score in this format:
Score: [number]
Reasoning: [brief explanation]"""

        try:
            # Call LLM judge
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.judge_temperature,
                max_tokens=500,
            )
            
            judge_response = response.choices[0].message.content
            
            # Parse score from response
            score_match = re.search(r'Score:\s*(\d+)', judge_response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))  # Clamp to 0-100
                normalized_score = score / 100.0
                
                logger.debug(f"Judge score: {score}/100")
                return normalized_score
            else:
                logger.warning(f"Could not parse score from judge response: {judge_response}")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in judge scoring: {e}")
            return 0.0


# ============================================================================
# Olympiads Prompt Builder Implementation
# ============================================================================

class OlympiadsPromptBuilder(PromptBuilder):
    """Prompt builder for Mathematical Olympiads benchmark"""
    
    def format_problem(self, problem: Problem) -> str:
        """Format the problem statement"""
        return f"Problem:\n{problem.question}"
    
    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        """Format a single attempt with its score"""
        content = attempt.output
        
        # Truncate to max_length tokens
        tokens = encoder.encode(content)
        if len(tokens) > max_length:
            truncated_tokens = tokens[-max_length:]
            content = encoder.decode(truncated_tokens)
            content = "..." + content
        
        # Replace multiple newlines with single newline
        content = re.sub(r'\n+', '\n', content)
        
        # Format with score (convert 0-1 reward to 0-100 score)
        score = int(attempt.reward * 100)
        formatted = f"<Previous Attempt>\n{content}\n</Previous Attempt>\n**Score: {score}/100**"
        
        return formatted
    
    def get_instruction(self, is_exploration: bool, attempts: List[Attempt]) -> str:
        """Get instruction based on round type and whether there are previous attempts"""
        
        # Initial attempts - no previous attempts to reference
        if not attempts:
            return "Please solve this problem step by step. Show your reasoning clearly and provide the final answer."
        
        # Subsequent attempts - reference previous attempts
        if is_exploration:
            return "Look at the previous attempts and their scores. Try a different approach or method to solve this problem."
        else:
            return "Look at the previous attempts and their scores. Provide the best possible solution, incorporating insights from high-scoring attempts."


# ============================================================================
# Logging Configuration
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    
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


def setup_logging(config: OlympiadsConfig):
    """Setup logging with color and file output"""
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
# Main Execution
# ============================================================================

def parse_config():
    """Parse configuration from command line and YAML"""
    default_config = OmegaConf.structured(OlympiadsConfig)
    
    # Check if a YAML config file is provided as first argument
    yaml_config = OmegaConf.create()
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        yaml_config = OmegaConf.load(sys.argv[1])
        # Remove the yaml file from argv so it's not parsed again
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
        
        # Save config
        with open(output_path / "config.yaml", "w") as f:
            OmegaConf.save(config, f)
    
    return config


async def main():
    """Main execution function"""
    # Parse configuration
    config = parse_config()
    
    # Setup logging
    setup_logging(config)
    
    logger.info("Starting Mathematical Olympiads ICRL benchmark")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Setup LLM clients
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    
    # Generation client
    generation_client = AsyncOpenAI(base_url=config.api_base, api_key=api_key)
    
    # Judge client (uses same or different model)
    judge_client = AsyncOpenAI(base_url=config.judge_api_base, api_key=api_key)
    
    # Create task with judge
    task = OlympiadsTask(
        config.dataset_name, 
        config.num_problems,
        judge_client=judge_client,
        judge_model_name=config.judge_model_name,
        judge_temperature=config.judge_temperature
    )
    
    # Create prompt builder
    prompt_builder = OlympiadsPromptBuilder()
    
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
    
    # Setup encoder (for length tracking)
    try:
        encoder = AutoTokenizer.from_pretrained(config.model_encoder)
    except Exception as e:
        logger.warning(f"Could not load tokenizer for {config.model_encoder}, using GPT-2 as fallback: {e}")
        encoder = AutoTokenizer.from_pretrained("gpt2")
    
    # Create ICRL config
    icrl_config = ICRLConfig(
        num_initial_attempts=config.num_initial_attempts,
        num_rounds=config.num_rounds,
        max_completion_tokens=config.max_completion_tokens,
        context_size=config.context_size,
        parallelization_degree=config.parallelization_degree,
        max_attempts_in_context=config.max_attempts_in_context,
        max_attempt_length=config.max_attempt_length,
        checkpoint_dir=config.output_dir if not config.debug_run else None,
    )
    
    # Create and run ICRL
    icrl = ICRL(
        task=task,
        prompt_builder=prompt_builder,
        llm_call=llm_call,
        encoder=encoder,
        config=icrl_config
    )
    
    # Solve
    final_state = await icrl.solve()
    
    # Report final results
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS")
    logger.info("="*100)
    
    all_best_rewards = []
    for i, ph in enumerate(final_state.problem_histories):
        if ph.attempts:
            best_reward = max(attempt.reward for attempt in ph.attempts)
            all_best_rewards.append(best_reward)
            if i < 5:  # Show first 5 problems
                logger.info(f"Problem {i}: Best reward = {best_reward:.3f} (score: {int(best_reward * 100)}/100)")
    
    if all_best_rewards:
        logger.info(f"\nOverall statistics:")
        logger.info(f"  Mean best reward: {sum(all_best_rewards) / len(all_best_rewards):.3f}")
        logger.info(f"  25th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//4]:.3f}")
        logger.info(f"  50th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//2]:.3f}")
        logger.info(f"  75th percentile: {sorted(all_best_rewards)[3*len(all_best_rewards)//4]:.3f}")
        logger.info(f"  Problems with perfect score (100/100): {sum(1 for r in all_best_rewards if r >= 0.99)}")
    
    logger.info("\nMathematical Olympiads ICRL benchmark complete!")


if __name__ == "__main__":
    anyio.run(main)

