"""
AIME Benchmark Implementation using ICRL Framework.
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, List

from openai import AsyncOpenAI
from datasets import load_dataset

from icrl import Task, Problem, Attempt, PromptBuilder
from benchmarks.common import (
    parse_config, setup_logging, make_llm_call, load_encoder, run_benchmark,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AIMEConfig:
    """Configuration for AIME benchmark"""
    # Dataset
    dataset_name: str = "AI-MO/aimo-validation-aime"
    num_problems: int = -1

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

    # Judge configuration
    judge_model_name: str = "gpt-4"
    judge_api_base: str = "https://api.openai.com/v1"
    judge_temperature: float = 0.0

    # Output
    base_output_dir: str = "ICL/aime/"
    postfix: str = ""
    output_dir: str = ""
    debug_run: bool = False


# ============================================================================
# Task
# ============================================================================

class AIMETask(Task):
    """Task implementation for AIME benchmark"""

    def __init__(self, dataset_name: str, num_problems: int = -1,
                 judge_client: AsyncOpenAI = None,
                 judge_model_name: str = "gpt-4",
                 judge_temperature: float = 0.0):
        self.dataset_name = dataset_name
        self.num_problems = num_problems
        self.judge_client = judge_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self._problems: Optional[List[Problem]] = None

    def get_problems(self) -> List[Problem]:
        if self._problems is not None:
            return self._problems

        dataset = load_dataset(self.dataset_name, split='train')
        if self.num_problems != -1:
            dataset = dataset.select(range(min(self.num_problems, len(dataset))))

        problems = []
        for data in dataset:
            problem = Problem(
                question=data['problem'],
                reference_answer=data['answer'],
                metadata={
                    'id': data['id'],
                    'solution': data['solution'],
                    'answer': data['answer'],
                    'url': data.get('url', ''),
                }
            )
            problems.append(problem)

        self._problems = problems
        logger.info(f"Loaded {len(problems)} problems from {self.dataset_name}")
        return problems

    async def score(self, solution: str, problem: Problem) -> float:
        reference_solution = problem.metadata['solution']
        reference_answer = problem.metadata['answer']

        judge_prompt = f"""You are an expert mathematics judge tasked with evaluating solutions to AIME (American Invitational Mathematics Examination) problems.

Problem:
{problem.question}

Reference Solution:
{reference_solution}

Reference Answer: {reference_answer}

Generated Solution:
{solution}

Instructions:
- Evaluate the generated solution based on the correctness of reasoning, approach, and mathematical validity.
- Give partial credit for partially correct solutions or valid approaches with minor errors.
- Give full marks (100) ONLY if the final answer is correct AND the reasoning is sound.
- If the final answer is incorrect, the maximum score should be less than 100, even if the approach has merit.

Provide a score from 0 to 100, where:
- 0 = Completely incorrect or no valid mathematical content
- 1-30 = Some relevant mathematical content but fundamentally flawed
- 31-60 = Partially correct approach with significant errors
- 61-90 = Mostly correct with minor errors or incomplete reasoning
- 91-99 = Nearly perfect but final answer is incorrect or minor issue
- 100 = Perfect solution with correct answer and sound reasoning

Respond with your score in this format:
Score: [number]
Reasoning: [brief explanation]"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.judge_model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.judge_temperature,
                max_tokens=500,
            )
            judge_response = response.choices[0].message.content
            score_match = re.search(r'Score:\s*(\d+)', judge_response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))
                return score / 100.0
            else:
                logger.warning(f"Could not parse score from judge response: {judge_response}")
                return 0.0
        except Exception as e:
            logger.warning(f"Error in judge scoring: {e}")
            return 0.0


# ============================================================================
# Prompt Builder
# ============================================================================

class AIMEPromptBuilder(PromptBuilder):
    """Prompt builder for AIME benchmark"""

    def format_problem(self, problem: Problem) -> str:
        return f"Problem:\n{problem.question}"

    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        content = attempt.output
        tokens = encoder.encode(content)
        if len(tokens) > max_length:
            truncated_tokens = tokens[-max_length:]
            content = encoder.decode(truncated_tokens)
            content = "..." + content
        content = re.sub(r'\n+', '\n', content)
        score = int(attempt.reward * 100)
        return f"<Previous Attempt>\n{content}\n</Previous Attempt>\n**Score: {score}/100**"

    def get_instruction(self, is_exploration: bool, attempts: List[Attempt]) -> str:
        if not attempts:
            return "Please solve this problem step by step. Show your reasoning clearly and provide the final answer."
        if is_exploration:
            return "Look at the previous attempts and their scores. Try a different approach or method to solve this problem."
        return "Look at the previous attempts and their scores. Provide the best possible solution, incorporating insights from high-scoring attempts."


# ============================================================================
# Main
# ============================================================================

async def main():
    config = parse_config(AIMEConfig)
    setup_logging(config)

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    generation_client = AsyncOpenAI(base_url=config.api_base, api_key=api_key)
    judge_client = AsyncOpenAI(base_url=config.judge_api_base, api_key=api_key)

    task = AIMETask(
        config.dataset_name, config.num_problems,
        judge_client=judge_client,
        judge_model_name=config.judge_model_name,
        judge_temperature=config.judge_temperature,
    )
    prompt_builder = AIMEPromptBuilder()
    llm_call = make_llm_call(generation_client, config.model_name, config.temperature, config.max_completion_tokens)
    encoder = load_encoder(config.model_encoder)

    await run_benchmark(config, task, prompt_builder, llm_call, encoder, "AIME", max_score_display=100)


if __name__ == "__main__":
    import anyio
    anyio.run(main)
