"""
Mathador Benchmark Implementation using ICRL Framework
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
import dotenv

# Import ICRL framework
from icrl import (
    Task, Problem, Attempt, PromptBuilder, 
    ICRL, ICRLConfig
)

# ============================================================================
# Mathador Scoring Functions (ported from Mathador/base.py)
# ============================================================================

def seval(l, r, op):
    """Evaluate a single operation safely"""
    if op == '+':
        return l + r, True
    elif op == '-':
        return l - r, l >= r
    elif op == '*':
        return l * r, True
    elif op == '/':
        return int(l / r) if r != 0 else None, r != 0 and l % r == 0
    else:
        raise ValueError(f'Invalid operator {op}')


def expr_to_shot(base_numbers, target, simple, simple_score, best, best_score):
    """Format a problem solution as a few-shot example"""
    simple_str = f"""Simple solution ({simple_score} points):
{simple}
"""
    best_str = f"""
Best solution ({best_score} points):
{best}
"""
    header = f"""Example:
Target number: {target}
Base numbers: {', '.join(map(str, base_numbers))}

"""
    if (not simple) or simple_score == best_score:
        return header + best_str
    else:
        return header + simple_str + best_str


def check_answer(message, target, base_numbers):
    """
    Check if a Mathador solution is correct and calculate its score.
    
    Args:
        message: The model's solution text
        target: The target number to reach
        base_numbers: List of available base numbers
        
    Returns:
        (score, reason): Score (0-18) and reason string
    """
    try:
        last_block = re.findall(r'((?:\s*(?:\n|^)\s*\d+\s*[+\-*\/]\s*\d+\s*=\s*\d+\s*)+)(?:\n|$)', message.strip())[-1]
    except:
        logger.debug('No answer block found')
        return 0, 'wrong_format'

    available_numbers = base_numbers.copy()
    score = 0
    used_operations = set()
    
    for line in last_block.strip().split('\n'):
        if line.isspace() or not line:
            continue
        try:
            oper1, operator, oper2, result = re.fullmatch(r'(\d+)\s*([+\-*\/])\s*(\d+)\s*=\s*(\d+)', line.strip()).groups()
        except:
            raise ValueError('This should not happen', line)
        
        try:
            if float(oper1) != int(float(oper1)) or float(oper2) != int(float(oper2)) or float(result) != int(float(result)):
                logger.debug('The numbers should be integers', line)
                return 0, 'illegal_intermediate_number'
        except OverflowError:
            logger.debug('The numbers are too big', line)
            return 0, 'illegal_intermediate_number'
        
        oper1, oper2, result = int(oper1), int(oper2), int(result)
        
        if oper1 < 0 or oper2 < 0 or result < 0:
            logger.debug('The numbers should be positive', line)
            return 0, 'illegal_intermediate_number'
        
        gold_result, valid = seval(oper1, oper2, operator)
        if gold_result != result or not valid:
            logger.debug('The calculation is not correct', line)
            return 0, 'wrong_calculation'
        
        try:
            available_numbers.remove(int(oper1))
            available_numbers.remove(int(oper2))
        except:
            logger.debug('You are using a number you should not', line)
            return 0, 'illegal_number_usage'
        
        available_numbers.append(int(result))

        if operator == '+':
            score += 1
        elif operator == '*':
            score += 1
        elif operator == '-':
            score += 2
        elif operator == '/':
            score += 3
        else:
            logger.debug('The operator is not valid', line)
            return 0, 'illegal_operator'

        used_operations.add(operator)

    if len(used_operations) == 4:
        score += 6

    assert score <= 13 or len(base_numbers) > 5

    if result != target:
        logger.debug('The result is not the target number')
        return 0, 'wrong_result'
    
    score += 5

    return score, 'correct'

dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MathadorConfig:
    """Configuration for Mathador benchmark"""
    # Dataset
    dataset_path: str = "Mathador/mathador-10000.jsonl"
    num_problems: int = -1  # -1 means all problems
    num_shots: int = 2  # Number of few-shot examples
    
    # ICRL parameters
    num_initial_attempts: int = 2
    num_rounds: int = 40
    max_completion_tokens: int = 4096
    context_size: int = 32768
    parallelization_degree: int = 10
    max_attempts_in_context: Optional[int] = None
    max_attempt_length: int = 512
    
    # Model configuration
    model_name: str = "gpt-4"
    model_encoder: str = "gpt-4"
    api_base: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 1.0
    
    # Output
    base_output_dir: str = "ICL/mathador/"
    postfix: str = ""
    output_dir: str = ""  # Will be set automatically with timestamp
    debug_run: bool = False


# ============================================================================
# Mathador Task Implementation
# ============================================================================

class MathadorTask(Task):
    """Task implementation for Mathador benchmark"""
    
    def __init__(self, dataset_path: str, num_problems: int = -1, num_shots: int = 2):
        """
        Initialize Mathador task.
        
        Args:
            dataset_path: Path to the jsonl dataset file
            num_problems: Number of problems to use (-1 for all)
            num_shots: Number of few-shot examples to include
        """
        self.dataset_path = dataset_path
        self.num_problems = num_problems
        self.num_shots = num_shots
        self._problems: Optional[List[Problem]] = None
    
    def get_problems(self) -> List[Problem]:
        """Load and return all problems from the dataset"""
        if self._problems is not None:
            return self._problems
        
        # Load entire dataset
        dataset = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        # Take shots from the end of the dataset (like eval.py does)
        shots_data = dataset[-self.num_shots:] if self.num_shots > 0 else []
        
        # Format shots as examples
        shots_text = ""
        if shots_data:
            shots_formatted = []
            for shot in shots_data:
                shot_example = expr_to_shot(
                    shot['base_numbers'],
                    shot['target'],
                    shot.get('simple_solution', ''),
                    shot.get('simple_solution_score', 0),
                    shot.get('mathador_solution', ''),
                    shot.get('mathador_solution_score', 0)
                )
                shots_formatted.append(shot_example)
            shots_text = '\n'.join(shots_formatted) + "\n\n"
        
        # Take problems from the beginning (excluding the shots from the end)
        problem_data = dataset[:len(dataset) - self.num_shots] if self.num_shots > 0 else dataset
        if self.num_problems != -1:
            problem_data = problem_data[:self.num_problems]
        
        # Create problem instances
        problems = []
        for data in problem_data:
            # Format the question using the proper Mathador prompt
            question = f"""Game description: In the Mathador game, players use the given base numbers and the operations of addition, subtraction, multiplication, and division to reach a specified target number.

Scoring:
- Each use of addition (+) is worth 1 point.
- Each use of multiplication (*) is worth 1 point.
- Each use of subtraction (-) is worth 2 points.
- Each use of division (/) is worth 3 points.
- 6 bonus points are awarded for using all four operations exactly once.

Rules:
- You should reach the target number.
- You should only use the base and intermediate numbers.
- You shouldn't use a base or intermediate number more than once in later steps.
- You should only produce nonnegative and integer intermediate results.
- Your solution should be 4 lines at most.

Only the solution you write at the end will be considered for scoring.
Find the highest scoring solution. If you are not able to find it, find a simple solution to earn at least some points.

{shots_text}Target number: {data['target']}
Base numbers: {', '.join(map(str, data['base_numbers']))}"""
            
            # Create problem instance
            problem = Problem(
                question=question,
                reference_answer=data['target'],
                metadata={
                    'target': data['target'],
                    'base_numbers': data['base_numbers'],
                    'mathador_solution': data.get('mathador_solution', ''),
                    'mathador_solution_score': data.get('mathador_solution_score', 0),
                }
            )
            problems.append(problem)
        
        self._problems = problems
        logger.info(f"Loaded {len(problems)} problems from {self.dataset_path}")
        if self.num_shots > 0:
            logger.info(f"Using {self.num_shots} few-shot examples in prompts")
        return problems
    
    async def score(self, solution: str, problem: Problem) -> float:
        """
        Score a solution using Mathador's check_answer function.
        
        Args:
            solution: The model's output
            problem: The problem being solved
            
        Returns:
            Normalized reward between 0 and 1
        """
        target = problem.metadata['target']
        base_numbers = problem.metadata['base_numbers']
        
        try:
            score, reason = check_answer(solution, target, base_numbers)
            # Normalize score to 0-1 range (max score is 18 for Mathador)
            normalized_score = score / 18.0
            return normalized_score
        except Exception as e:
            logger.warning(f"Error scoring solution: {e}")
            return 0.0


# ============================================================================
# Mathador Prompt Builder Implementation
# ============================================================================

class MathadorPromptBuilder(PromptBuilder):
    """Prompt builder for Mathador benchmark"""
    
    def format_problem(self, problem: Problem) -> str:
        """Format the problem statement"""
        return problem.question
    
    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        """Format a single attempt with its score"""
        content = attempt.output
        
        # Truncate to max_length tokens
        tokens = encoder.encode(content)
        if len(tokens) > max_length:
            truncated_tokens = tokens[-max_length:] + encoder.encode("...")
            content = encoder.decode(truncated_tokens)
        
        # Replace multiple newlines with single newline
        content = re.sub(r'\n+', '\n', content)
        
        # Format with score (convert 0-1 reward to 0-18 score)
        score = int(attempt.reward * 18)
        formatted = f"<Attempt>\n{content}\n</Attempt>\n**Score achieved: {score}/18 points**"
        
        return formatted
    
    def get_instruction(self, is_exploration: bool, attempts: List[Attempt]) -> str:
        """Get instruction based on round type and whether there are previous attempts"""
        
        # Initial attempts - no previous attempts to reference
        if not attempts:
            return """Please provide your solution now. The formatting should exactly follow the examples in the question to allow automatic scoring."""
        
        # Subsequent attempts - reference previous attempts
        if is_exploration:
            return """Look at the previous attempts and their scores. Try to construct a new solution that is different from all of them. The formatting should exactly follow the examples in the question to allow automatic scoring."""
        else:
            return """Look at the previous attempts and their scores. Try to construct a solution that scores high. The formatting should exactly follow the examples in the question to allow automatic scoring."""


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


def setup_logging(config: MathadorConfig):
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


# ============================================================================
# Main Execution
# ============================================================================

def parse_config():
    """Parse configuration from command line and YAML"""
    default_config = OmegaConf.structured(MathadorConfig)
    
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
    
    logger.info("Starting Mathador ICRL benchmark")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Create task
    task = MathadorTask(config.dataset_path, config.num_problems, config.num_shots)
    
    # Create prompt builder
    prompt_builder = MathadorPromptBuilder()
    
    # Setup LLM client
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=config.api_base, api_key=api_key)
    
    # Create LLM call wrapper
    async def llm_call(messages: list[dict]) -> str:
        """Wrapper for LLM API calls"""
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_completion_tokens,
        )
        return response.choices[0].message.content
    
    # Setup encoder (for length tracking)
    # Use the actual tokenizer for accurate token counting
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
                logger.info(f"Problem {i}: Best reward = {best_reward:.3f} (score: {int(best_reward * 18)}/18)")
    
    logger.info(f"\nOverall statistics:")
    logger.info(f"  Mean best reward: {sum(all_best_rewards) / len(all_best_rewards):.3f}")
    logger.info(f"  25th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//4]:.3f}")
    logger.info(f"  50th percentile: {sorted(all_best_rewards)[len(all_best_rewards)//2]:.3f}")
    logger.info(f"  75th percentile: {sorted(all_best_rewards)[3*len(all_best_rewards)//4]:.3f}")
    logger.info(f"  Problems with perfect score (18/18): {sum(1 for r in all_best_rewards if r >= 0.99)}")
    
    logger.info("\nMathador ICRL benchmark complete!")


if __name__ == "__main__":
    anyio.run(main)

