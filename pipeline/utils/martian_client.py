"""
Martian API Client Wrapper

Wrapper around OpenAI SDK configured for Martian's API endpoint.
Handles judge evaluation with structured JSON responses, retry logic, and rate limit handling.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, RootModel
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


class SingleDimensionScore(BaseModel):
    """Single dimension evaluation with score and explanation."""
    score: float = Field(description="Numerical score")
    explanation: str = Field(description="Brief explanation (2-3 sentences)")


class DimensionScore(BaseModel):
    """Score and explanation for one dimension in multi-dimensional evaluation."""
    score: float = Field(description="Numerical score")
    explanation: str = Field(description="Brief explanation (1-2 sentences)")


class MultiDimensionEvaluation(RootModel[Dict[str, DimensionScore]]):
    """Multi-dimensional evaluation with dynamic dimension names."""
    root: Dict[str, DimensionScore]


class MartianClient:
    """Client for Martian API using OpenAI SDK with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.withmartian.com/v1",
        default_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        timeout: float = 60.0
    ):
        """
        Initialize Martian client.

        Args:
            api_key: Martian API key (defaults to MARTIAN_API_KEY env var)
            base_url: Martian API base URL
            default_model: Default model to use (with provider prefix)
            temperature: Default temperature for sampling
            max_retries: Maximum retry attempts for failed requests
            initial_retry_delay: Initial delay in seconds for exponential backoff
            max_retry_delay: Maximum delay between retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("MARTIAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MARTIAN_API_KEY not found. Set it in .env or pass explicitly."
            )

        self.base_url = base_url
        self.default_model = default_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.timeout = timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )

        logger.info(f"Initialized Martian client with base_url: {base_url}")

    def _calculate_retry_delay(self, attempt: int, base_delay: Optional[float] = None) -> float:
        """
        Calculate delay for exponential backoff with jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)
            base_delay: Optional custom base delay (uses initial_retry_delay if None)

        Returns:
            Delay in seconds with exponential backoff and jitter
        """
        import random

        base = base_delay or self.initial_retry_delay
        # Exponential backoff: base * 2^attempt
        delay = min(base * (2 ** attempt), self.max_retry_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    def evaluate_with_rubric(
        self,
        rubric: str,
        question: str,
        answer: str,
        model: Optional[str] = None,
        multi_dimensional: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a Q&A pair using a judge rubric with retry logic.

        Args:
            rubric: Judge rubric prompt (system message)
            question: The question/instruction to evaluate
            answer: The response to evaluate
            model: Model to use (defaults to default_model)
            multi_dimensional: If True, use MultiDimensionEvaluation format

        Returns:
            Parsed evaluation dict
        """
        model = model or self.default_model

        user_message = f"""Evaluate the following response:

Question/Instruction:
{question}

Response to Evaluate:
{answer}

Provide your evaluation following the rubric criteria."""

        text_format = MultiDimensionEvaluation if multi_dimensional else SingleDimensionScore
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": user_message}
                    ],
                    text_format=text_format,
                    temperature=self.temperature
                )

                evaluation = response.output_parsed

                if multi_dimensional:
                    return evaluation.root
                else:
                    return {"score": evaluation.score, "explanation": evaluation.explanation}

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Extract retry-after header if available
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = float(retry_after)
                        logger.warning(f"Rate limited. Waiting {delay}s (from Retry-After header)")
                    else:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Rate limited. Waiting {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue

            except (APITimeoutError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Connection/timeout error: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue

            except APIError as e:
                last_exception = e
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                    if attempt < self.max_retries - 1:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Server error {e.status_code}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                raise

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Unexpected error: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                raise

        logger.error(f"Evaluation failed after {self.max_retries} attempts")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Evaluation failed after {self.max_retries} attempts with unknown error")

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, str]],
        model: Optional[str] = None,
        max_workers: int = 5,
        multi_dimensional: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple Q&A pairs in parallel using ThreadPoolExecutor.

        Args:
            evaluations: List of dicts with 'rubric', 'question', 'answer'
            model: Model to use
            max_workers: Number of parallel workers (reasonable: 5-20)
            multi_dimensional: If True, expect multi-dimensional scores

        Returns:
            List of evaluation results (same order as input)
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
                    model,
                    multi_dimensional
                ): idx
                for idx, eval_dict in enumerate(evaluations)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch evaluation {idx} failed after retries: {e}")
                    results[idx] = {"score": 2.0, "explanation": f"Error: {e}"}

        return results


def load_client() -> MartianClient:
    """
    Load Martian client from environment configuration.

    Returns:
        Initialized MartianClient
    """
    return MartianClient()
