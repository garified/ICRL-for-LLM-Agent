"""
Novelty scorer for Mathador benchmark.

Uses mutual information (MI) between solutions, approximated via LM perplexity,
to reward diverse exploration. Solutions that are genuinely different from
previous attempts receive a higher novelty score.

MI(s_new, s_prev) = log2(PP_uncond(s_new)) - log2(PP_cond(s_new | s_prev))
novelty_reward = max(0, 1 - mean_MI)
"""

import asyncio
import logging
import threading
from functools import partial

import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)


def calculate_perplexity(context_text, solution_text, model, tokenizer, device="cuda"):
    """Calculate conditional and unconditional perplexity of a solution.

    Args:
        context_text: The context/conditioning text
        solution_text: The solution text to evaluate
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on

    Returns:
        Tuple of (ppx_conditional, ppx_unconditional)
    """
    context_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=True)
    solution_tokens = tokenizer("\n\n" + solution_text, return_tensors="pt", add_special_tokens=False)

    context_ids = context_tokens.input_ids.to(device)
    solution_ids = solution_tokens.input_ids.to(device)

    # Conditional perplexity: P(solution | context)
    input_ids = torch.cat([context_ids, solution_ids], dim=-1)
    labels = torch.cat([
        torch.full_like(context_ids, -100),
        solution_ids,
    ], dim=-1)

    with torch.no_grad():
        outputs_cond = model(input_ids=input_ids, labels=labels)
        ppx_cond = torch.exp(outputs_cond.loss).item()

    # Unconditional perplexity: P(solution)
    with torch.no_grad():
        outputs_uncond = model(input_ids=solution_ids, labels=solution_ids)
        ppx_uncond = torch.exp(outputs_uncond.loss).item()

    return ppx_cond, ppx_uncond


class NoveltyScorer:
    """Scores solution novelty using mutual information approximated via LM perplexity."""

    def __init__(self, model_name: str, device: str = "cuda"):
        logger.info(f"Loading novelty model: {model_name} on {device}")
        self.device = device
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._lock = threading.Lock()
        logger.info("Novelty model loaded")

    def calculate_mi(self, context: str, text: str) -> float:
        """Calculate mutual information I(text; context) = log2(PP_uncond) - log2(PP_cond).

        Args:
            context: The conditioning text (e.g., a previous solution)
            text: The text to measure (e.g., the new solution)

        Returns:
            Mutual information in bits
        """
        with self._lock:
            ppx_cond, ppx_uncond = calculate_perplexity(
                context, text, self.model, self.tokenizer, self.device
            )
        return np.log2(ppx_uncond) - np.log2(ppx_cond)

    def score_novelty(self, solution: str, previous_solutions: list[str]) -> float:
        """Score the novelty of a solution relative to previous solutions.

        Args:
            solution: The new solution to evaluate
            previous_solutions: List of previous solution strings

        Returns:
            Novelty score in [0, 1]. Higher means more novel.
        """
        if not previous_solutions:
            return 1.0

        mi_values = [self.calculate_mi(prev, solution) for prev in previous_solutions]
        mean_mi = np.mean(mi_values)
        return max(0.0, 1.0 - mean_mi)

    async def score_novelty_async(self, solution: str, previous_solutions: list[str]) -> float:
        """Async wrapper for score_novelty to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self.score_novelty, solution, previous_solutions)
        )
