"""
Tests for the novelty reward integration in the Mathador benchmark.

Covers:
1. icrl/__init__.py — score() returning float vs tuple[float, dict]
2. benchmarks/mathador/novelty.py — NoveltyScorer (mocked torch)
3. benchmarks/mathador/__init__.py — MathadorTask.score() with/without novelty
4. benchmarks/mathador/__init__.py — MathadorPromptBuilder.format_attempt()
"""

import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import field

from icrl import Attempt, Problem, Task, PromptBuilder, ICRL, ICRLConfig


# ============================================================================
# 1. ICRL core: score() tuple handling
# ============================================================================


class FloatScoreTask(Task):
    """Task that returns a plain float from score()."""

    def get_problems(self):
        return [Problem(question="test", reference_answer="ans")]

    async def score(self, solution, problem):
        return 0.75


class TupleScoreTask(Task):
    """Task that returns a tuple from score()."""

    def get_problems(self):
        return [Problem(question="test", reference_answer="ans")]

    async def score(self, solution, problem):
        return (0.85, {"task_reward": 0.7, "novelty_reward": 0.5})


class DummyPromptBuilder(PromptBuilder):
    def format_problem(self, problem):
        return problem.question

    def format_attempt(self, attempt, encoder, max_length):
        return attempt.output

    def get_instruction(self, is_exploration, attempts):
        return "solve"


class TestICRLScoreTupleHandling(unittest.TestCase):
    """Test that ICRL handles both float and tuple returns from score()."""

    def _make_icrl(self, task):
        encoder = MagicMock()
        encoder.encode = MagicMock(return_value=[1, 2, 3])
        llm_call = AsyncMock(return_value="solution text")
        config = ICRLConfig(
            num_initial_attempts=1,
            num_rounds=1,
            checkpoint_dir=None,
        )
        return ICRL(task, DummyPromptBuilder(), llm_call, encoder, config)

    def test_float_score_sets_empty_extra_fields(self):
        """When score() returns float, extra_fields should be {}."""
        icrl = self._make_icrl(FloatScoreTask())
        state = asyncio.run(icrl.solve())

        # Should have attempts from initial + 1 round
        ph = state.problem_histories[0]
        self.assertTrue(len(ph.attempts) >= 1)

        for attempt in ph.attempts:
            self.assertEqual(attempt.extra_fields, {})
            self.assertAlmostEqual(attempt.reward, 0.75)

    def test_tuple_score_populates_extra_fields(self):
        """When score() returns tuple, extra_fields should be populated."""
        icrl = self._make_icrl(TupleScoreTask())
        state = asyncio.run(icrl.solve())

        ph = state.problem_histories[0]
        self.assertTrue(len(ph.attempts) >= 1)

        for attempt in ph.attempts:
            self.assertAlmostEqual(attempt.reward, 0.85)
            self.assertIn("task_reward", attempt.extra_fields)
            self.assertIn("novelty_reward", attempt.extra_fields)
            self.assertAlmostEqual(attempt.extra_fields["task_reward"], 0.7)
            self.assertAlmostEqual(attempt.extra_fields["novelty_reward"], 0.5)


# ============================================================================
# 2. NoveltyScorer (mocked torch)
# ============================================================================


class TestNoveltyScorer(unittest.TestCase):
    """Test NoveltyScorer logic with mocked model."""

    def _make_scorer(self):
        """Create a NoveltyScorer with mocked model loading."""
        with patch("benchmarks.mathador.novelty.transformers") as mock_tf:
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_tf.AutoModelForCausalLM.from_pretrained.return_value.to.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = MagicMock()

            from benchmarks.mathador.novelty import NoveltyScorer
            scorer = NoveltyScorer("fake-model", device="cpu")

        return scorer

    def test_no_previous_solutions_returns_1(self):
        """With no previous solutions, novelty should be 1.0."""
        scorer = self._make_scorer()
        result = scorer.score_novelty("new solution", [])
        self.assertAlmostEqual(result, 1.0)

    def test_high_mi_gives_low_novelty(self):
        """When MI is high (similar solutions), novelty should be low."""
        scorer = self._make_scorer()
        # Mock calculate_mi to return high MI (similar)
        scorer.calculate_mi = MagicMock(return_value=0.8)
        result = scorer.score_novelty("new", ["prev1", "prev2"])
        # novelty = max(0, 1 - 0.8) = 0.2
        self.assertAlmostEqual(result, 0.2)

    def test_low_mi_gives_high_novelty(self):
        """When MI is low (different solutions), novelty should be high."""
        scorer = self._make_scorer()
        scorer.calculate_mi = MagicMock(return_value=0.1)
        result = scorer.score_novelty("new", ["prev1"])
        # novelty = max(0, 1 - 0.1) = 0.9
        self.assertAlmostEqual(result, 0.9)

    def test_very_high_mi_clamps_to_zero(self):
        """When MI > 1, novelty should clamp to 0."""
        scorer = self._make_scorer()
        scorer.calculate_mi = MagicMock(return_value=2.5)
        result = scorer.score_novelty("new", ["prev1"])
        self.assertAlmostEqual(result, 0.0)

    def test_mean_mi_over_multiple_previous(self):
        """MI should be averaged over all previous solutions."""
        scorer = self._make_scorer()
        scorer.calculate_mi = MagicMock(side_effect=[0.2, 0.6, 0.4])
        result = scorer.score_novelty("new", ["p1", "p2", "p3"])
        # mean MI = (0.2 + 0.6 + 0.4) / 3 = 0.4
        # novelty = max(0, 1 - 0.4) = 0.6
        self.assertAlmostEqual(result, 0.6)
        self.assertEqual(scorer.calculate_mi.call_count, 3)

    def test_score_novelty_async(self):
        """Async wrapper should return same result."""
        scorer = self._make_scorer()
        scorer.calculate_mi = MagicMock(return_value=0.3)
        result = asyncio.run(scorer.score_novelty_async("new", ["prev"]))
        self.assertAlmostEqual(result, 0.7)


# ============================================================================
# 3. MathadorTask.score() with and without novelty
# ============================================================================


class TestMathadorTaskScore(unittest.TestCase):
    """Test MathadorTask scoring with/without novelty."""

    def _make_task(self, novelty_reward=False, novelty_weight=0.3):
        """Create a MathadorTask, patching NoveltyScorer if novelty enabled."""
        from benchmarks.mathador import MathadorTask

        if novelty_reward:
            with patch("benchmarks.mathador.novelty.transformers") as mock_tf:
                mock_model = MagicMock()
                mock_model.eval = MagicMock()
                mock_tf.AutoModelForCausalLM.from_pretrained.return_value.to.return_value = mock_model
                mock_tf.AutoTokenizer.from_pretrained.return_value = MagicMock()

                task = MathadorTask(
                    dataset_path="Mathador/mathador-10000.jsonl",
                    num_problems=2,
                    num_shots=0,
                    novelty_reward=True,
                    novelty_weight=novelty_weight,
                    novelty_model="fake-model",
                    novelty_device="cpu",
                )
        else:
            task = MathadorTask(
                dataset_path="Mathador/mathador-10000.jsonl",
                num_problems=2,
                num_shots=0,
                novelty_reward=False,
            )
        return task

    def test_without_novelty_returns_float(self):
        """Without novelty, score() should return a plain float."""
        task = self._make_task(novelty_reward=False)
        problems = task.get_problems()
        problem = problems[0]

        # Use a known valid solution: "5 + 20 = 25" for target=25
        solution = "5 + 20 = 25"
        result = asyncio.run(task.score(solution, problem))

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_with_novelty_returns_tuple(self):
        """With novelty, score() should return (combined_reward, extra_fields)."""
        task = self._make_task(novelty_reward=True, novelty_weight=0.3)
        problems = task.get_problems()
        problem = problems[0]

        # Mock the novelty scorer
        task._novelty_scorer.score_novelty_async = AsyncMock(return_value=0.8)

        solution = "5 + 20 = 25"
        result = asyncio.run(task.score(solution, problem))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        combined_reward, extra = result
        self.assertIn("task_reward", extra)
        self.assertIn("novelty_reward", extra)
        self.assertAlmostEqual(extra["novelty_reward"], 0.8)

        # combined = 0.7 * task_reward + 0.3 * 0.8
        expected = 0.7 * extra["task_reward"] + 0.3 * 0.8
        self.assertAlmostEqual(combined_reward, expected, places=5)

    def test_novelty_tracks_solutions(self):
        """Novelty scoring should track solutions per problem."""
        task = self._make_task(novelty_reward=True)
        problems = task.get_problems()
        problem = problems[0]

        task._novelty_scorer.score_novelty_async = AsyncMock(return_value=1.0)

        asyncio.run(task.score("sol1", problem))
        asyncio.run(task.score("sol2", problem))

        key = hash(problem.question)
        self.assertEqual(len(task._solutions[key]), 2)
        self.assertEqual(task._solutions[key], ["sol1", "sol2"])

    def test_invalid_solution_returns_zero_task_reward(self):
        """Invalid solution should give task_reward=0, novelty still computed."""
        task = self._make_task(novelty_reward=True, novelty_weight=0.3)
        problems = task.get_problems()
        problem = problems[0]

        task._novelty_scorer.score_novelty_async = AsyncMock(return_value=0.9)

        result = asyncio.run(task.score("garbage nonsense", problem))
        combined, extra = result
        self.assertAlmostEqual(extra["task_reward"], 0.0)
        self.assertAlmostEqual(extra["novelty_reward"], 0.9)
        # combined = 0.7 * 0.0 + 0.3 * 0.9 = 0.27
        self.assertAlmostEqual(combined, 0.27, places=5)


# ============================================================================
# 4. MathadorPromptBuilder.format_attempt()
# ============================================================================


class TestMathadorFormatAttempt(unittest.TestCase):
    """Test format_attempt with and without novelty extra_fields."""

    def _get_encoder(self):
        encoder = MagicMock()
        encoder.encode = MagicMock(return_value=[1, 2, 3])
        encoder.decode = MagicMock(return_value="decoded")
        return encoder

    def test_format_without_novelty(self):
        """Without extra_fields, should show standard score line."""
        from benchmarks.mathador import MathadorPromptBuilder

        builder = MathadorPromptBuilder()
        attempt = Attempt(
            prompt=[],
            output="5 + 20 = 25",
            reward=6 / 18.0,
            round_idx=0,
        )
        result = builder.format_attempt(attempt, self._get_encoder(), 512)
        self.assertIn("Score achieved: 6/18 points", result)
        self.assertNotIn("Novelty", result)

    def test_format_with_novelty(self):
        """With novelty extra_fields, should show both task score and novelty."""
        from benchmarks.mathador import MathadorPromptBuilder

        builder = MathadorPromptBuilder()
        attempt = Attempt(
            prompt=[],
            output="20 * 5 = 100\n100 - 4 = 96\n96 / 4 = 24\n24 + 1 = 25",
            reward=0.85,  # combined reward
            round_idx=0,
            extra_fields={"task_reward": 1.0, "novelty_reward": 0.65},
        )
        result = builder.format_attempt(attempt, self._get_encoder(), 512)
        self.assertIn("Task score: 18/18 points", result)
        self.assertIn("Novelty: 0.65", result)
        self.assertNotIn("Score achieved", result)


# ============================================================================
# 5. calculate_perplexity (mocked torch)
# ============================================================================


class TestCalculatePerplexity(unittest.TestCase):
    """Test the perplexity calculation function with mocked torch."""

    @patch("benchmarks.mathador.novelty.torch")
    def test_returns_cond_and_uncond(self, mock_torch):
        """Should return (ppx_cond, ppx_uncond) tuple."""
        from benchmarks.mathador.novelty import calculate_perplexity

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(
            input_ids=MagicMock(to=MagicMock(return_value=MagicMock()))
        )

        # Mock model outputs
        mock_loss = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(loss=mock_loss)

        # Mock torch operations
        mock_torch.cat = MagicMock(return_value=MagicMock())
        mock_torch.full_like = MagicMock(return_value=MagicMock())
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.exp = MagicMock(return_value=MagicMock(item=MagicMock(side_effect=[3.5, 7.2])))

        ppx_cond, ppx_uncond = calculate_perplexity(
            "context", "solution", mock_model, mock_tokenizer, "cpu"
        )

        self.assertEqual(ppx_cond, 3.5)
        self.assertEqual(ppx_uncond, 7.2)


if __name__ == "__main__":
    unittest.main()
