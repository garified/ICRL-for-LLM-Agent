"""
ICRL Framework - A modular framework for In-Context Reinforcement Learning
"""

import pickle
import re
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable
from pathlib import Path
import anyio

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Problem:
    """Base problem data structure"""
    question: str
    reference_answer: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class Attempt:
    """Stores a single attempt with its metadata"""
    prompt: list[dict]
    output: str
    reward: float
    round_idx: int
    extra_fields: dict = field(default_factory=dict)


@dataclass
class ProblemHistory:
    """Stores a problem and all its attempts"""
    problem: Problem
    attempts: List[Attempt] = field(default_factory=list)


# ============================================================================
# Abstract Base Classes
# ============================================================================

class Task(ABC):
    """Abstract base class for tasks to be solved with ICRL"""
    
    @abstractmethod
    def get_problems(self) -> List[Problem]:
        """Return list of all problems to solve"""
        pass
    
    @abstractmethod
    async def score(self, solution: str, problem: Problem) -> float:
        """
        Score a solution attempt.
        
        Args:
            solution: The model's output/solution
            problem: The problem being solved
            
        Returns:
            Reward value between 0 and 1
        """
        pass


class PromptBuilder(ABC):
    """
    Abstract base class for building prompts for ICRL.
    
    Users only need to implement three methods:
    1. format_problem() - How to present the problem
    2. format_attempt() - How to display a single attempt
    3. get_instruction() - What instruction to give (exploration vs exploitation)
    
    The framework handles all the boilerplate: length tracking, sorting attempts,
    truncation, merging messages, etc.
    """
    
    @abstractmethod
    def format_problem(self, problem: Problem) -> str:
        """
        Format the problem statement for display.
        
        Args:
            problem: The problem to format
            
        Returns:
            Formatted string representation of the problem
        """
        pass
    
    @abstractmethod
    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        """
        Format a single attempt for display in prompt.
        
        Args:
            attempt: The attempt to format
            encoder: Tokenizer for length calculations
            max_length: Maximum token length for the formatted attempt
            
        Returns:
            Formatted string representation of the attempt
        """
        pass
    
    @abstractmethod
    def get_instruction(self, is_exploration: bool, attempts: List[Attempt]) -> str:
        """
        Get the instruction for the current round.
        
        Args:
            is_exploration: True for exploration rounds, False for exploitation
            attempts: List of previous attempts (empty for initial attempts)
            
        Returns:
            Instruction text to append to the prompt
        """
        pass
    
    def build_prompt(
        self, 
        problem: Problem, 
        attempts: List[Attempt], 
        is_exploration: bool,
        encoder,
        max_length: int,
        max_attempts_in_context: Optional[int] = None,
        max_attempt_length: int = 512
    ) -> list[dict]:
        """
        Build full prompt messages from problem and past attempts.
        
        This is a concrete method that handles all the standard logic:
        - Length tracking to fit within context
        - Sorting attempts by reward (best first)
        - Limiting number of attempts
        - Truncating long attempts
        - Merging consecutive user messages
        
        Users don't override this - they implement format_problem(), format_attempt(), 
        and get_instruction() instead.
        
        Args:
            problem: The problem to solve
            attempts: List of previous attempts
            is_exploration: Whether this is an exploration or exploitation round
            encoder: Tokenizer for length tracking
            max_length: Maximum total token length for the prompt
            max_attempts_in_context: Maximum number of attempts to include
            max_attempt_length: Maximum token length for each individual attempt
            
        Returns:
            List of message dicts in OpenAI format [{"role": "user", "content": "..."}]
        """
        messages = []
        length_tracker = LengthTracker(max_length, encoder)
        
        # Prepare instruction message (check it fits)
        instruction = self.get_instruction(is_exploration, attempts)
        instruction_message = {"role": "user", "content": f"\n\n{instruction}"}
        if not length_tracker.can_i_add_this_message(instruction_message):
            raise ValueError("Instruction message is too long for context!")
        
        # Add problem statement
        problem_text = self.format_problem(problem)
        problem_message = {"role": "user", "content": f"{problem_text}\n\n"}
        if not length_tracker.can_i_add_this_message(problem_message):
            raise ValueError("Problem message is too long for context!")
        messages.append(problem_message)
        
        # Sort attempts by reward (highest first) and limit if specified
        sorted_attempts = sorted(attempts, key=lambda x: x.reward, reverse=True)
        if max_attempts_in_context is not None:
            sorted_attempts = sorted_attempts[:max_attempts_in_context]
        
        # Add attempts until we run out of space
        for i, attempt in enumerate(sorted_attempts):
            formatted_attempt = self.format_attempt(attempt, encoder, max_attempt_length)
            
            # Add spacing between attempts (except for the first one)
            if i > 0:
                formatted_attempt = "\n\n" + formatted_attempt
            
            message = {"role": "user", "content": formatted_attempt}
            if not length_tracker.can_i_add_this_message(message):
                break  # No more space
            messages.append(message)
        
        # Add instruction at the end
        messages.append(instruction_message)
        
        # Merge consecutive user messages
        messages = merge_same_role_messages(messages)
        
        return messages


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ICRLConfig:
    """Configuration for ICRL framework"""
    # Experiment parameters
    num_initial_attempts: int = 2
    num_rounds: int = 40
    max_completion_tokens: int = 4096
    context_size: int = 32768
    context_size_safety_margin: int = 75
    max_attempt_length: int = 512
    
    # Execution parameters
    parallelization_degree: int = 10
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    
    # Context management
    max_attempts_in_context: Optional[int] = None


# ============================================================================
# Utility Classes
# ============================================================================

class LengthTracker:
    """Tracks token length to ensure prompts fit within context limits"""
    
    def __init__(self, length_limit: int, encoder, safety_margin: int = 75):
        self.length_limit = length_limit - safety_margin
        self.current_length = 0
        self.encoder = encoder
    
    def can_i_add_this_message(self, message: dict) -> bool:
        """Check if adding a message would exceed the length limit"""
        new_text = message['role'] + ": " + message['content']
        new_tokens = len(self.encoder.encode(new_text))
        if self.current_length + new_tokens > self.length_limit:
            return False
        self.current_length += new_tokens
        return True


def merge_same_role_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role"""
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages


# ============================================================================
# Main ICRL Class
# ============================================================================

@dataclass
class ICRLState:
    """Stores the complete state of an ICRL run for checkpointing"""
    problem_histories: List[ProblemHistory] = field(default_factory=list)
    config: ICRLConfig = None
    
    def save(self, filepath: Path):
        """Save state to disk"""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'ICRLState':
        """Load state from disk"""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        logger.info(f"Loaded checkpoint from {filepath}")
        return state


class ICRL:
    """
    Main ICRL orchestration class.
    
    This class handles:
    - Initial exploration phase
    - Iterative rounds of exploration and exploitation
    - Checkpointing and resumption
    - Parallel execution across problems
    """
    
    def __init__(
        self,
        task: Task,
        prompt_builder: PromptBuilder,
        llm_call: Callable,
        encoder,
        config: ICRLConfig
    ):
        """
        Initialize ICRL framework.
        
        Args:
            task: Task implementation (defines problems and scoring)
            prompt_builder: PromptBuilder implementation (defines prompt formatting)
            llm_call: async function(messages: list[dict]) -> str that calls the LLM
            encoder: Tokenizer for length tracking (should have .encode() method)
            config: ICRL configuration
        """
        self.task = task
        self.prompt_builder = prompt_builder
        self.llm_call = llm_call
        self.encoder = encoder
        self.config = config
        self.state: Optional[ICRLState] = None
    
    def _get_checkpoint_path(self, round_idx: int) -> Path:
        """Get checkpoint path for a given round"""
        if self.config.checkpoint_dir is None:
            return None
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if round_idx == -1:
            return checkpoint_dir / "checkpoint_initial.pkl"
        return checkpoint_dir / f"checkpoint_round_{round_idx}.pkl"
    
    def _load_or_initialize_state(self) -> ICRLState:
        """Load state from checkpoint or initialize new state"""
        if self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                # Find the latest checkpoint
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
                if checkpoints:
                    return ICRLState.load(checkpoints[-1])
        
        # Initialize new state
        problems = self.task.get_problems()
        state = ICRLState(
            problem_histories=[ProblemHistory(problem=p) for p in problems],
            config=self.config
        )
        return state
    
    async def _run_initial_attempts(self):
        """Run initial exploration attempts for all problems"""
        logger.info(f"Running {self.config.num_initial_attempts} initial attempts for {len(self.state.problem_histories)} problems")
        
        async def initial_interaction(problem_idx: int):
            """Run initial attempts for a single problem"""
            problem_history = self.state.problem_histories[problem_idx]
            problem = problem_history.problem
            
            for _ in range(self.config.num_initial_attempts):
                # Build initial prompt using prompt builder (empty attempts list)
                problem_text = self.prompt_builder.format_problem(problem)
                # Pass empty list for initial attempts - user can check if attempts is empty
                initial_instruction = self.prompt_builder.get_instruction(
                    is_exploration=True,  # Initial attempts are exploratory
                    attempts=[]
                )
                
                messages = [{
                    "role": "user", 
                    "content": f"{problem_text}\n\n{initial_instruction}"
                }]
                
                # Get model output
                model_output = await self.llm_call(messages)
                
                # Score the attempt
                reward = await self.task.score(model_output, problem)
                
                # Store the attempt
                attempt = Attempt(
                    prompt=messages,
                    output=model_output,
                    reward=reward,
                    round_idx=-1
                )
                problem_history.attempts.append(attempt)
                
                if problem_idx == 0:
                    logger.info(f"\n{'-'*100}\nInitial attempt (problem 0): {model_output[:200]}...\n{'-'*100}\n")
        
        # Run in parallel with limited concurrency
        async with anyio.create_task_group() as tg:
            semaphore = anyio.Semaphore(self.config.parallelization_degree)
            
            async def run_with_semaphore(idx):
                async with semaphore:
                    await initial_interaction(idx)
            
            for i in range(len(self.state.problem_histories)):
                tg.start_soon(run_with_semaphore, i)
        
        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(-1)
        if checkpoint_path:
            self.state.save(checkpoint_path)
        
        # Log statistics
        rewards = [
            attempt.reward 
            for ph in self.state.problem_histories 
            for attempt in ph.attempts 
            if attempt.round_idx == -1
        ]
        if rewards:
            logger.info(
                f"Initial rewards - "
                f"25th: {np.percentile(rewards, 25):.3f}, "
                f"50th: {np.percentile(rewards, 50):.3f}, "
                f"75th: {np.percentile(rewards, 75):.3f}"
            )
    
    async def _run_round(self, round_idx: int):
        """Run a single ICRL round for all problems"""
        is_exploration = (round_idx % 2 == 0)
        round_type = "exploration" if is_exploration else "exploitation"
        logger.info(f"Running round {round_idx} ({round_type})")
        
        async def round_interaction(problem_idx: int):
            """Run one round iteration for a single problem"""
            problem_history = self.state.problem_histories[problem_idx]
            problem = problem_history.problem
            
            # Calculate available context length
            available_length = (
                self.config.context_size 
                - self.config.max_completion_tokens 
                - self.config.context_size_safety_margin
            )
            
            # Build prompt using the prompt builder
            messages = self.prompt_builder.build_prompt(
                problem=problem,
                attempts=problem_history.attempts,
                is_exploration=is_exploration,
                encoder=self.encoder,
                max_length=available_length,
                max_attempts_in_context=self.config.max_attempts_in_context,
                max_attempt_length=self.config.max_attempt_length
            )
            
            # Get model output
            model_output = await self.llm_call(messages)
            
            # Score the attempt
            reward = await self.task.score(model_output, problem)
            
            # Store the attempt
            attempt = Attempt(
                prompt=messages,
                output=model_output,
                reward=reward,
                round_idx=round_idx
            )
            problem_history.attempts.append(attempt)
            
            if problem_idx == 0:
                logger.info(f"\n{'-'*100}\nRound {round_idx} attempt (problem 0): {model_output[:200]}...\n{'-'*100}\n")
        
        # Run in parallel with limited concurrency
        async with anyio.create_task_group() as tg:
            semaphore = anyio.Semaphore(self.config.parallelization_degree)
            
            async def run_with_semaphore(idx):
                async with semaphore:
                    await round_interaction(idx)
            
            for i in range(len(self.state.problem_histories)):
                tg.start_soon(run_with_semaphore, i)
        
        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(round_idx)
        if checkpoint_path:
            # Delete previous checkpoint to save space
            if round_idx > 0:
                prev_checkpoint = self._get_checkpoint_path(round_idx - 1)
                if prev_checkpoint and prev_checkpoint.exists():
                    prev_checkpoint.unlink()
            elif round_idx == 0:
                initial_checkpoint = self._get_checkpoint_path(-1)
                if initial_checkpoint and initial_checkpoint.exists():
                    initial_checkpoint.unlink()
            
            self.state.save(checkpoint_path)
        
        # Log statistics
        rewards = [
            attempt.reward 
            for ph in self.state.problem_histories 
            for attempt in ph.attempts 
            if attempt.round_idx == round_idx
        ]
        logger.info(
            f"Round {round_idx} rewards - "
            f"25th: {np.percentile(rewards, 25):.3f}, "
            f"50th: {np.percentile(rewards, 50):.3f}, "
            f"75th: {np.percentile(rewards, 75):.3f}"
        )
    
    def _get_start_round(self) -> int:
        """Determine which round to start from based on existing attempts"""
        start_round = 0
        for problem_history in self.state.problem_histories:
            if problem_history.attempts:
                max_round = max(attempt.round_idx for attempt in problem_history.attempts)
                start_round = max(start_round, max_round + 1)
        return start_round
    
    async def solve(self) -> ICRLState:
        """
        Main solve loop - runs initial attempts + all rounds.
        
        Returns:
            Final ICRLState with all problem histories
        """
        # Load or initialize state
        self.state = self._load_or_initialize_state()
        
        # Determine starting point
        start_round = self._get_start_round()
        
        # Run initial attempts if not done yet
        if start_round <= 0 and self.config.num_initial_attempts > 0:
            needs_initial = any(
                len(ph.attempts) < self.config.num_initial_attempts 
                for ph in self.state.problem_histories
            )
            if needs_initial:
                await self._run_initial_attempts()
                start_round = 0
        
        # Run iterative rounds
        for round_idx in range(start_round, self.config.num_rounds):
            await self._run_round(round_idx)
        
        logger.info("ICRL solving complete!")
        return self.state


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Problem',
    'Attempt',
    'ProblemHistory',
    'Task',
    'PromptBuilder',
    'ICRLConfig',
    'ICRLState',
    'ICRL',
    'LengthTracker',
    'merge_same_role_messages',
]

