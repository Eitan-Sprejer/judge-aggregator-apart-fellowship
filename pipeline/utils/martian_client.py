"""
Martian API Client Wrapper

Wrapper around OpenAI SDK configured for Martian's API endpoint.
Handles judge evaluation with structured JSON responses, retry logic, and rate limit handling.
"""

import os
import re
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
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
    """Client for Martian API using OpenAI SDK with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.withmartian.com/v1",
        default_model: str = "openai/gpt-4o-mini",
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
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.timeout = timeout

        # Initialize OpenAI client with Martian endpoint
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
        temperature: float = 0.0,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a Q&A pair using a judge rubric with retry logic.

        Automatically retries on:
        - Rate limit errors (429)
        - Timeout errors
        - Connection errors
        - Temporary API errors (5xx)

        Args:
            rubric: Judge rubric prompt (system message)
            question: The question/instruction to evaluate
            answer: The response to evaluate
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature (default 0.0 for deterministic)
            max_retries: Override default max_retries

        Returns:
            Dictionary with 'score' (float) and 'explanation' (str)

        Raises:
            Exception: If evaluation fails after max_retries
        """
        model = model or self.default_model
        max_retries = max_retries or self.max_retries

        # Construct user message with Q&A to evaluate
        user_message = f"""Evaluate the following response:

Question/Instruction:
{question}

Response to Evaluate:
{answer}

Provide your evaluation following the rubric criteria."""

        last_exception = None

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                # Use the correct responses.parse API with 'text_format' parameter
                response = self.client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": user_message}
                    ],
                    text_format=JudgeEvaluation,
                    temperature=temperature
                )

                # Extract parsed response - use output_parsed attribute
                evaluation = response.output_parsed

                if evaluation is None:
                    # Fallback: try to parse from text content if available
                    logger.warning("Structured parsing failed, attempting manual parse")
                    content = getattr(response, 'text', '') or str(response)
                    return self._parse_fallback(content)

                return {
                    "score": evaluation.score,
                    "explanation": evaluation.explanation
                }

            except RateLimitError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Extract retry-after header if available
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = float(retry_after)
                        logger.warning(f"Rate limited. Waiting {delay}s (from Retry-After header)")
                    else:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Rate limited. Waiting {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

            except (APITimeoutError, APIConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Connection/timeout error: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue

            except APIError as e:
                last_exception = e
                # Retry on 5xx server errors, not 4xx client errors (except 429 handled above)
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600:
                    if attempt < max_retries - 1:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Server error {e.status_code}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                # Don't retry client errors (4xx except 429)
                raise

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Unexpected error: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise

        # If we exhausted retries, raise the last exception
        logger.error(f"Evaluation failed after {max_retries} attempts")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Evaluation failed after {max_retries} attempts with unknown error")

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
        evaluations: List[Dict[str, str]],
        model: Optional[str] = None,
        max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple Q&A pairs in parallel using ThreadPoolExecutor.

        Note: ThreadPoolExecutor is appropriate here because:
        1. API calls are I/O-bound (waiting for network responses)
        2. Python's GIL doesn't affect I/O-bound operations
        3. Simpler than async/await for this use case
        4. Works well with existing retry logic

        For very high throughput (>100 concurrent requests), consider
        using asyncio with httpx instead.

        Args:
            evaluations: List of dicts with 'rubric', 'question', 'answer'
            model: Model to use
            max_workers: Number of parallel workers (reasonable: 5-20)

        Returns:
            List of evaluation results (same order as input)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(evaluations)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
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

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch evaluation {idx} failed after retries: {e}")
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
