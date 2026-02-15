"""
Mathador Benchmark Implementation using ICRL Framework.

Scoring logic is imported from Mathador.base (no duplication).
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional, List

from openai import AsyncOpenAI

from icrl import Task, Problem, Attempt, PromptBuilder
from Mathador.base import seval, expr_to_shot, check_answer
from benchmarks.common import (
    parse_config, setup_logging, make_llm_call, load_encoder, run_benchmark,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MathadorConfig:
    """Configuration for Mathador benchmark"""
    # Dataset
    dataset_path: str = "Mathador/mathador-10000.jsonl"
    num_problems: int = -1
    num_shots: int = 2

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

    # Novelty reward
    novelty_reward: bool = False
    novelty_weight: float = 0.3
    novelty_model: str = "Qwen/Qwen3-0.6B"
    novelty_device: str = "cuda"

    # Output
    base_output_dir: str = "ICL/mathador/"
    postfix: str = ""
    output_dir: str = ""
    debug_run: bool = False


# ============================================================================
# Task
# ============================================================================

class MathadorTask(Task):
    """Task implementation for Mathador benchmark"""

    def __init__(
        self,
        dataset_path: str,
        num_problems: int = -1,
        num_shots: int = 2,
        novelty_reward: bool = False,
        novelty_weight: float = 0.3,
        novelty_model: str = "Qwen/Qwen3-0.6B",
        novelty_device: str = "cuda",
    ):
        self.dataset_path = dataset_path
        self.num_problems = num_problems
        self.num_shots = num_shots
        self._problems: Optional[List[Problem]] = None

        self._novelty_enabled = novelty_reward
        self._novelty_weight = novelty_weight
        self._novelty_scorer = None
        self._solutions: dict[str, list[str]] = {}

        if self._novelty_enabled:
            from benchmarks.mathador.novelty import NoveltyScorer
            self._novelty_scorer = NoveltyScorer(novelty_model, novelty_device)

    def get_problems(self) -> List[Problem]:
        if self._problems is not None:
            return self._problems

        dataset = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))

        # Take shots from the end of the dataset
        shots_data = dataset[-self.num_shots:] if self.num_shots > 0 else []

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
                    shot.get('mathador_solution_score', 0),
                )
                shots_formatted.append(shot_example)
            shots_text = '\n'.join(shots_formatted) + "\n\n"

        problem_data = dataset[:len(dataset) - self.num_shots] if self.num_shots > 0 else dataset
        if self.num_problems != -1:
            problem_data = problem_data[:self.num_problems]

        problems = []
        for data in problem_data:
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

    async def score(self, solution: str, problem: Problem):
        target = problem.metadata['target']
        base_numbers = problem.metadata['base_numbers']
        try:
            raw_score, reason = check_answer(solution, target, base_numbers)
            task_reward = raw_score / 18.0
        except Exception as e:
            logger.warning(f"Error scoring solution: {e}")
            task_reward = 0.0

        if not self._novelty_enabled:
            return task_reward

        # Compute novelty reward
        problem_key = hash(problem.question)
        prev_solutions = self._solutions.get(problem_key, [])
        novelty = await self._novelty_scorer.score_novelty_async(solution, prev_solutions)

        # Track this solution for future comparisons
        self._solutions.setdefault(problem_key, []).append(solution)

        combined = (1 - self._novelty_weight) * task_reward + self._novelty_weight * novelty
        return (combined, {'task_reward': task_reward, 'novelty_reward': novelty})


# ============================================================================
# Prompt Builder
# ============================================================================

class MathadorPromptBuilder(PromptBuilder):
    """Prompt builder for Mathador benchmark"""

    def format_problem(self, problem: Problem) -> str:
        return problem.question

    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        content = attempt.output
        tokens = encoder.encode(content)
        if len(tokens) > max_length:
            truncated_tokens = tokens[-max_length:] + encoder.encode("...")
            content = encoder.decode(truncated_tokens)
        content = re.sub(r'\n+', '\n', content)
        if 'novelty_reward' in attempt.extra_fields:
            novelty = attempt.extra_fields['novelty_reward']
            task_score = int(attempt.extra_fields['task_reward'] * 18)
            score_line = f"**Task score: {task_score}/18 points | Novelty: {novelty:.2f}**"
        else:
            score = int(attempt.reward * 18)
            score_line = f"**Score achieved: {score}/18 points**"
        return f"<Attempt>\n{content}\n</Attempt>\n{score_line}"

    def get_instruction(self, is_exploration: bool, attempts: List[Attempt]) -> str:
        if not attempts:
            return "Please provide your solution now. The formatting should exactly follow the examples in the question to allow automatic scoring."
        if is_exploration:
            return "Look at the previous attempts and their scores. Try to construct a new solution that is different from all of them. The formatting should exactly follow the examples in the question to allow automatic scoring."
        return "Look at the previous attempts and their scores. Try to construct a solution that scores high. The formatting should exactly follow the examples in the question to allow automatic scoring."


# ============================================================================
# Main
# ============================================================================

async def main():
    config = parse_config(MathadorConfig)
    setup_logging(config)

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=config.api_base, api_key=api_key)

    task = MathadorTask(
        config.dataset_path, config.num_problems, config.num_shots,
        novelty_reward=config.novelty_reward,
        novelty_weight=config.novelty_weight,
        novelty_model=config.novelty_model,
        novelty_device=config.novelty_device,
    )
    prompt_builder = MathadorPromptBuilder()
    llm_call = make_llm_call(client, config.model_name, config.temperature, config.max_completion_tokens)
    encoder = load_encoder(config.model_encoder)

    await run_benchmark(config, task, prompt_builder, llm_call, encoder, "Mathador", max_score_display=18)


if __name__ == "__main__":
    import anyio
    anyio.run(main)
