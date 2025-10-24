"""
Martian API Client Wrapper

Simple wrapper around OpenAI SDK configured to use Martian's API endpoint.
Handles judge evaluation with structured JSON responses.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


class JudgeEvaluation(BaseModel):
    """Structured output format for judge evaluations."""
    explanation: str = Field(
        description="Brief explanation of the evaluation reasoning (2-3 sentences)"
    )
    score: float = Field(
        description="Numerical score from 0.0 to 4.0",
        ge=0.0,
        le=4.0
    )


class MartianClient:
    """Client for Martian API using OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.withmartian.com/v1",
        default_model: str = "openai/gpt-4o-mini"
    ):
        """
        Initialize Martian client.

        Args:
            api_key: Martian API key (defaults to MARTIAN_API_KEY env var)
            base_url: Martian API base URL
            default_model: Default model to use (with provider prefix)
        """
        self.api_key = api_key or os.getenv("MARTIAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MARTIAN_API_KEY not found. Set it in .env or pass explicitly."
            )

        self.base_url = base_url
        self.default_model = default_model

        # Initialize OpenAI client with Martian endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"Initialized Martian client with base_url: {base_url}")

    def evaluate_with_rubric(
        self,
        rubric: str,
        question: str,
        answer: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a Q&A pair using a judge rubric.

        Args:
            rubric: Judge rubric prompt (system message)
            question: The question/instruction to evaluate
            answer: The response to evaluate
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature (default 0.0 for deterministic)
            max_retries: Maximum retry attempts on failure

        Returns:
            Dictionary with 'score' (float) and 'explanation' (str)

        Raises:
            Exception: If evaluation fails after max_retries
        """
        model = model or self.default_model

        # Construct user message with Q&A to evaluate
        user_message = f"""Evaluate the following response:

Question/Instruction:
{question}

Response to Evaluate:
{answer}

Provide your evaluation following the rubric criteria."""

        # Make API call with structured output
        for attempt in range(max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": user_message}
                    ],
                    response_format=JudgeEvaluation,
                    temperature=temperature
                )

                # Extract parsed response
                evaluation = response.choices[0].message.parsed

                if evaluation is None:
                    # Fallback: try to parse from content
                    content = response.choices[0].message.content
                    logger.warning("Structured parsing failed, attempting manual parse")
                    return self._parse_fallback(content)

                return {
                    "score": evaluation.score,
                    "explanation": evaluation.explanation
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                else:
                    logger.error(f"Evaluation failed after {max_retries} attempts: {e}")
                    raise

    def _parse_fallback(self, content: str) -> Dict[str, Any]:
        """
        Fallback parser for when structured output fails.

        Attempts to extract score and explanation from free-form text.

        Args:
            content: Raw response content

        Returns:
            Dictionary with 'score' and 'explanation'
        """
        # Try to find score with regex patterns
        score_patterns = [
            r'"?score"?\s*:\s*(\d+\.?\d*)',  # JSON format
            r'score:\s*(\d+\.?\d*)',  # Plain text
            r'rating:\s*(\d+\.?\d*)',  # Alternative
            r'(\d+\.?\d*)\s*/\s*4',  # "3.5 / 4" format
        ]

        score = 2.0  # Default mid-range score
        for pattern in score_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to valid range
                    score = max(0.0, min(4.0, score))
                    break
                except ValueError:
                    continue

        # Use first sentence or paragraph as explanation
        explanation = content.split('\n')[0][:200]  # Limit length

        logger.info(f"Fallback parse extracted score: {score}")
        return {"score": score, "explanation": explanation}

    def evaluate_batch(
        self,
        evaluations: list[Dict[str, str]],
        model: Optional[str] = None,
        max_workers: int = 5
    ) -> list[Dict[str, Any]]:
        """
        Evaluate multiple Q&A pairs in parallel.

        Args:
            evaluations: List of dicts with 'rubric', 'question', 'answer'
            model: Model to use
            max_workers: Number of parallel workers

        Returns:
            List of evaluation results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(evaluations)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.evaluate_with_rubric,
                    eval_dict["rubric"],
                    eval_dict["question"],
                    eval_dict["answer"],
                    model
                ): idx
                for idx, eval_dict in enumerate(evaluations)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch evaluation {idx} failed: {e}")
                    results[idx] = {"score": 2.0, "explanation": f"Error: {e}"}

        return results


def load_client(config_path: Optional[str] = None) -> MartianClient:
    """
    Load Martian client from environment configuration.

    Args:
        config_path: Optional path to config file (currently unused, for compatibility)

    Returns:
        Initialized MartianClient
    """
    return MartianClient()
