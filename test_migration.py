"""
Test script for Martian SDK → OpenAI SDK migration

Tests:
1. Client initialization
2. Judge loading
3. Single evaluation
4. JSON response parsing
"""

import logging
from pipeline.utils.martian_client import MartianClient
from pipeline.core.judge_creation import get_judge_ids, get_judge_info
from pipeline.core.judge_evaluation import JudgeEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_client_initialization():
    """Test 1: Verify client can be initialized"""
    logger.info("\n=== Test 1: Client Initialization ===")
    try:
        client = MartianClient()
        logger.info("✅ Client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Client initialization failed: {e}")
        return False


def test_judge_loading():
    """Test 2: Verify judges can be loaded"""
    logger.info("\n=== Test 2: Judge Loading ===")
    try:
        judge_ids = get_judge_ids()
        logger.info(f"Available judges: {len(judge_ids)}")

        # Test loading first judge
        first_judge = judge_ids[0]
        judge_info = get_judge_info(first_judge)

        logger.info(f"✅ Loaded judge: {first_judge}")
        logger.info(f"   Description: {judge_info['description']}")
        logger.info(f"   Rubric length: {len(judge_info['rubric'])} chars")
        return True
    except Exception as e:
        logger.error(f"❌ Judge loading failed: {e}")
        return False


def test_single_evaluation():
    """Test 3: Perform single evaluation"""
    logger.info("\n=== Test 3: Single Evaluation ===")
    try:
        # Initialize evaluator with just one judge
        judge_ids = get_judge_ids()
        first_judge = judge_ids[0]

        evaluator = JudgeEvaluator(judge_ids=[first_judge])

        # Simple test Q&A
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."

        # Evaluate
        logger.info(f"Evaluating with judge: {first_judge}")
        score = evaluator.evaluate_single(question, answer, first_judge)

        logger.info(f"✅ Evaluation successful")
        logger.info(f"   Score: {score}")
        logger.info(f"   Valid range: {0.0 <= score <= 4.0}")

        if not (0.0 <= score <= 4.0):
            logger.warning(f"⚠️ Score {score} out of valid range [0.0, 4.0]")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_evaluation():
    """Test 4: Parallel evaluation with multiple judges"""
    logger.info("\n=== Test 4: Parallel Evaluation ===")
    try:
        # Initialize evaluator with 3 judges
        judge_ids = get_judge_ids()[:3]

        evaluator = JudgeEvaluator(judge_ids=judge_ids)

        # Simple test Q&A
        question = "Explain quantum computing in simple terms."
        answer = "Quantum computing uses quantum bits that can be in multiple states at once, allowing for more powerful computations than traditional computers."

        # Evaluate
        logger.info(f"Evaluating with {len(judge_ids)} judges in parallel")
        scores = evaluator.evaluate_parallel(question, answer, max_workers=3)

        logger.info(f"✅ Parallel evaluation successful")
        for judge_id, score in zip(judge_ids, scores):
            logger.info(f"   {judge_id}: {score}")

        # Verify all scores valid
        all_valid = all(0.0 <= score <= 4.0 for score in scores)
        if not all_valid:
            logger.warning(f"⚠️ Some scores out of valid range")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Parallel evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Martian SDK Migration Test Suite")
    logger.info("=" * 60)

    tests = [
        ("Client Initialization", test_client_initialization),
        ("Judge Loading", test_judge_loading),
        ("Single Evaluation", test_single_evaluation),
        ("Parallel Evaluation", test_parallel_evaluation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("🎉 All tests passed! Migration successful.")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
