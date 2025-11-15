# ICRL Framework

A modular framework for In-Context Reinforcement Learning (ICRL) that enables iterative improvement of LLM solutions through exploration and exploitation.

## Overview

The ICRL framework provides a clean abstraction for applying in-context reinforcement learning to any task. It handles:

- **Initial exploration** phase with random attempts
- **Iterative rounds** alternating between exploration (trying new approaches) and exploitation (improving best attempts)
- **Context management** to fit prompts within token limits
- **Parallel execution** across multiple problems
- **Checkpointing** for resuming interrupted runs
- **Logging and metrics** tracking progress

## Architecture

The framework consists of three main components:

### 1. Task (Abstract Base Class)

Defines the problems to solve and how to score solutions.

```python
class Task(ABC):
    @abstractmethod
    def get_problems(self) -> List[Problem]:
        """Return list of all problems to solve"""
        pass
    
    @abstractmethod
    async def score(self, solution: str, problem: Problem) -> float:
        """Score a solution, returns reward between 0 and 1"""
        pass
```

### 2. PromptBuilder (Abstract Base Class)

Defines how to format prompts for the LLM, including how to present previous attempts.

```python
class PromptBuilder(ABC):
    @abstractmethod
    def format_attempt(self, attempt: Attempt, encoder, max_length: int) -> str:
        """Format a single attempt for display"""
        pass
    
    @abstractmethod
    def get_exploration_instruction(self) -> str:
        """Instruction for exploration rounds"""
        pass
    
    @abstractmethod
    def get_exploitation_instruction(self) -> str:
        """Instruction for exploitation rounds"""
        pass
    
    @abstractmethod
    def build_prompt(self, problem: Problem, attempts: List[Attempt], 
                     is_exploration: bool, encoder, max_length: int,
                     max_attempts_in_context: Optional[int]) -> list[dict]:
        """Build full prompt with problem and past attempts"""
        pass
```

### 3. ICRL (Main Orchestrator)

Coordinates the entire ICRL process.

```python
class ICRL:
    def __init__(self, task: Task, prompt_builder: PromptBuilder,
                 llm_call: Callable, encoder, config: ICRLConfig):
        """
        Args:
            task: Task implementation
            prompt_builder: PromptBuilder implementation
            llm_call: async function(messages) -> str for LLM calls
            encoder: Tokenizer with .encode() and .decode() methods
            config: ICRL configuration
        """
    
    async def solve(self) -> ICRLState:
        """Run ICRL and return final state with all attempts"""
```

## Usage Example

See `mathador_bench.py` for a complete example. Here's the basic pattern:

```python
from icrl import Task, Problem, PromptBuilder, ICRL, ICRLConfig

# 1. Implement Task
class MyTask(Task):
    def get_problems(self):
        # Load and return your problems
        return [Problem(question="...", reference_answer="...")]
    
    async def score(self, solution, problem):
        # Score the solution (0 to 1)
        return 0.8

# 2. Implement PromptBuilder
class MyPromptBuilder(PromptBuilder):
    def format_attempt(self, attempt, encoder, max_length):
        return f"<Attempt>\n{attempt.output}\n</Attempt>\nScore: {attempt.reward}"
    
    def get_exploration_instruction(self):
        return "Try a completely different approach..."
    
    def get_exploitation_instruction(self):
        return "Improve on the previous best attempts..."
    
    def build_prompt(self, problem, attempts, is_exploration, 
                     encoder, max_length, max_attempts_in_context):
        # Build the full prompt
        messages = [{"role": "user", "content": problem.question}]
        # Add formatted attempts...
        return messages

# 3. Create LLM call function
async def llm_call(messages):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content

# 4. Run ICRL
config = ICRLConfig(
    num_initial_attempts=2,
    num_rounds=20,
    max_completion_tokens=4096,
    parallelization_degree=10,
    checkpoint_dir="results/my_task"
)

icrl = ICRL(
    task=MyTask(),
    prompt_builder=MyPromptBuilder(),
    llm_call=llm_call,
    encoder=tokenizer,
    config=config
)

final_state = await icrl.solve()
```

## Configuration

The `ICRLConfig` dataclass controls all aspects of the ICRL process:

```python
@dataclass
class ICRLConfig:
    # Experiment parameters
    num_initial_attempts: int = 2       # Random attempts before ICRL
    num_rounds: int = 40                # Number of ICRL rounds
    max_completion_tokens: int = 4096   # Max tokens for LLM output
    context_size: int = 32768           # Total context window size
    
    # Execution parameters
    parallelization_degree: int = 10    # Number of concurrent problems
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None  # Where to save checkpoints
    
    # Context management
    max_attempts_in_context: Optional[int] = None  # Limit attempts in prompt
    max_attempt_length: int = 512       # Max tokens per attempt display
```

## Data Structures

### Problem
```python
@dataclass
class Problem:
    question: str              # The problem statement
    reference_answer: Any      # Ground truth answer
    metadata: dict            # Additional problem-specific data
```

### Attempt
```python
@dataclass
class Attempt:
    prompt: list[dict]        # Messages sent to LLM
    output: str               # LLM's response
    reward: float             # Score (0 to 1)
    round_idx: int            # Which round (-1 for initial)
    extra_fields: dict        # Additional data
```

### ICRLState
```python
@dataclass
class ICRLState:
    problem_histories: List[ProblemHistory]  # All problems and attempts
    config: ICRLConfig                       # Configuration used
```

## Checkpointing

ICRL automatically saves checkpoints after each round if `checkpoint_dir` is set:

- `checkpoint_initial.pkl` - After initial attempts
- `checkpoint_round_0.pkl`, `checkpoint_round_1.pkl`, etc. - After each round

To resume from a checkpoint, simply run with the same `checkpoint_dir`. The framework will automatically detect and load the latest checkpoint.

## Utilities

The framework provides several utility functions:

- `LengthTracker`: Track token counts to stay within context limits
- `merge_same_role_messages()`: Combine consecutive messages with same role

## Best Practices

1. **Token Management**: Use `LengthTracker` in your `build_prompt()` to ensure prompts fit
2. **Attempt Formatting**: Truncate long attempts to save context space
3. **Scoring**: Normalize scores to 0-1 range for consistent behavior
4. **Parallelization**: Adjust `parallelization_degree` based on API rate limits
5. **Checkpointing**: Always use checkpoints for long runs

## Implementation Notes

- Exploration rounds (even indices) encourage diversity
- Exploitation rounds (odd indices) encourage improvement
- Attempts are sorted by reward (highest first) when building prompts
- Old checkpoints are automatically deleted to save space
- All problem interactions run in parallel (up to `parallelization_degree`)

## See Also

- `mathador_bench.py` - Complete implementation for Mathador benchmark
- `math_bench.py` - Original implementation (monolithic, for reference)

