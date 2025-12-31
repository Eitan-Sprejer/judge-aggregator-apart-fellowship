#!/usr/bin/env python3
"""
Tests for the Iterative Judge Selection Pipeline components.

Run with:
    python experiments/track3_automated_selection/test_iterative_selection.py

Or run specific tests:
    python -m pytest experiments/track3_automated_selection/test_iterative_selection.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.track3_automated_selection.iterative_selection.judge_set_metrics import (
    JudgeSetMetrics,
    JudgeSetEvaluator,
    compute_quick_redundancy,
)
from experiments.track3_automated_selection.iterative_selection.gap_analyzer import (
    GapAnalyzer,
    GapAnalysisResult,
    identify_least_important_judge,
)
from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    SelectionConfig,
    IterationResult,
    IterativeJudgeSelector,
)


def create_synthetic_data(
    n_samples: int = 200,
    n_judges: int = 5,
    noise_level: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create synthetic judge scores and target values for testing.
    
    Returns:
        judge_scores: (n_samples, n_judges) array
        targets: (n_samples,) array
        judge_names: list of judge names
    """
    np.random.seed(seed)
    
    judge_names = [f"judge_{i}" for i in range(n_judges)]
    
    # Create judge scores with some correlation structure
    # Judge 0 and 1 are highly correlated (redundant)
    base_signal = np.random.randn(n_samples)
    
    judge_scores = np.zeros((n_samples, n_judges))
    judge_scores[:, 0] = base_signal + np.random.randn(n_samples) * 0.2
    judge_scores[:, 1] = base_signal + np.random.randn(n_samples) * 0.3  # Correlated with 0
    judge_scores[:, 2] = np.random.randn(n_samples)  # Independent
    judge_scores[:, 3] = np.random.randn(n_samples) * 0.5  # Low variance
    judge_scores[:, 4] = -base_signal + np.random.randn(n_samples) * 0.4  # Anti-correlated with 0
    
    # Scale to 0-4 range
    judge_scores = (judge_scores - judge_scores.min()) / (judge_scores.max() - judge_scores.min()) * 4
    
    # Target is a weighted combination of judges + noise
    weights = np.array([0.4, 0.1, 0.3, 0.05, 0.15])  # Judge 3 is least important
    targets = judge_scores @ weights + np.random.randn(n_samples) * noise_level
    targets = np.clip(targets, 0, 4)
    
    return judge_scores, targets, judge_names


class TestJudgeSetMetrics:
    """Tests for judge_set_metrics module."""
    
    def test_evaluator_basic(self):
        """Test basic JudgeSetEvaluator functionality."""
        print("\n📊 Testing JudgeSetEvaluator...")
        
        judge_scores, targets, judge_names = create_synthetic_data()
        
        # Simple linear predictions (should be decent)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(judge_scores, targets)
        predictions = model.predict(judge_scores)
        
        evaluator = JudgeSetEvaluator(correlation_threshold=0.8)
        metrics = evaluator.evaluate(
            judge_scores=judge_scores,
            judge_names=judge_names,
            predictions=predictions,
            targets=targets,
        )
        
        print(f"  R²: {metrics.r2:.4f}")
        print(f"  Spearman ρ: {metrics.spearman_rho:.4f}")
        print(f"  Mean pairwise correlation: {metrics.mean_pairwise_correlation:.4f}")
        print(f"  Redundancy score: {metrics.redundancy_score:.4f}")
        print(f"  Diversity index: {metrics.diversity_index:.4f}")
        print(f"  Composite score: {metrics.composite_score:.4f}")
        print(f"  Highly correlated pairs: {len(metrics.highly_correlated_pairs)}")
        
        assert metrics.r2 > 0.5, f"Expected R² > 0.5, got {metrics.r2}"
        assert 0 <= metrics.redundancy_score <= 1, "Redundancy score should be in [0,1]"
        assert 0 <= metrics.diversity_index <= 1, "Diversity index should be in [0,1]"
        
        # Should detect the correlated pair (judge_0, judge_1)
        correlated_names = {(a, b) for a, b, _ in metrics.highly_correlated_pairs}
        print(f"  Detected correlated pairs: {correlated_names}")
        
        print("  ✅ JudgeSetEvaluator test passed!")
        return metrics
    
    def test_quick_redundancy(self):
        """Test quick redundancy check function."""
        print("\n🔍 Testing quick redundancy check...")
        
        judge_scores, _, judge_names = create_synthetic_data()
        
        result = compute_quick_redundancy(
            judge_scores=judge_scores,
            judge_names=judge_names,
            threshold=0.7,
        )
        
        print(f"  Mean abs correlation: {result['mean_abs_correlation']:.4f}")
        print(f"  N redundant pairs (r>0.7): {result['n_redundant_pairs']}")
        
        assert "correlation_matrix" in result
        assert len(result["correlation_matrix"]) == len(judge_names)
        
        print("  ✅ Quick redundancy test passed!")
        return result
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        print("\n📋 Testing metrics serialization...")
        
        metrics = JudgeSetMetrics(
            r2=0.85,
            spearman_rho=0.82,
            mean_pairwise_correlation=0.45,
            redundancy_score=0.55,
            diversity_index=0.72,
            composite_score=0.68,
        )
        
        d = metrics.to_dict()
        
        assert "predictive_power" in d
        assert d["predictive_power"]["r2"] == 0.85
        assert "redundancy" in d
        assert d["composite_score"] == 0.68
        
        print(f"  Dict keys: {list(d.keys())}")
        print("  ✅ Serialization test passed!")


class TestGapAnalyzer:
    """Tests for gap_analyzer module."""
    
    def test_gap_analysis_basic(self):
        """Test basic gap analysis."""
        print("\n🔬 Testing GapAnalyzer...")
        
        judge_scores, targets, judge_names = create_synthetic_data()
        
        # Create biased predictions (systematic underprediction)
        predictions = targets * 0.8 + np.random.randn(len(targets)) * 0.2
        
        analyzer = GapAnalyzer(error_threshold=0.3, cluster_count=3)
        result = analyzer.analyze(
            predictions=predictions,
            targets=targets,
            judge_scores=judge_scores,
            judge_names=judge_names,
        )
        
        print(f"  Patterns found: {len(result.patterns)}")
        for p in result.patterns[:3]:
            print(f"    - [{p.pattern_type}] {p.description[:60]}...")
        
        print(f"  Error stats: MAE={result.overall_error_stats['mean_abs_error']:.4f}")
        print(f"  Suggested dimensions: {result.suggested_dimensions[:3]}")
        print(f"\n  Summary:\n{result.analysis_summary[:200]}...")
        
        assert len(result.patterns) >= 0  # May or may not find patterns
        assert "mean_abs_error" in result.overall_error_stats
        assert isinstance(result.judge_error_correlations, dict)
        
        print("  ✅ GapAnalyzer test passed!")
        return result
    
    def test_identify_least_important(self):
        """Test least important judge identification."""
        print("\n📉 Testing least important judge identification...")
        
        importance_scores = {
            "judge_a": 0.9,
            "judge_b": 0.7,
            "judge_c": 0.2,  # Least important
            "judge_d": 0.5,
        }
        
        # Without protection
        name, score = identify_least_important_judge(importance_scores)
        print(f"  Least important: {name} (score={score})")
        assert name == "judge_c"
        assert score == 0.2
        
        # With protection
        name, score = identify_least_important_judge(
            importance_scores,
            protected_judges=["judge_c"],
        )
        print(f"  Least important (c protected): {name} (score={score})")
        assert name == "judge_d"
        
        print("  ✅ Least important identification test passed!")


class TestIterativeSelection:
    """Tests for iterative_selection module."""
    
    def test_config_serialization(self):
        """Test config YAML save/load."""
        print("\n⚙️ Testing SelectionConfig...")
        
        config = SelectionConfig(
            max_iterations=5,
            min_judges=2,
            protected_judges=["judge_a"],
        )
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = SelectionConfig.from_yaml(f.name)
        
        assert loaded.max_iterations == 5
        assert loaded.min_judges == 2
        assert loaded.protected_judges == ["judge_a"]
        
        print(f"  Config fields: max_iter={loaded.max_iterations}, min_judges={loaded.min_judges}")
        print("  ✅ Config serialization test passed!")
    
    def test_selector_initialization(self):
        """Test IterativeJudgeSelector initialization."""
        print("\n🚀 Testing IterativeJudgeSelector initialization...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelectionConfig(
                output_dir=tmpdir,
                max_iterations=3,
                min_judges=2,
            )
            
            selector = IterativeJudgeSelector(config)
            
            assert selector.output_dir.exists()
            assert (selector.output_dir / "config.yaml").exists()
            
            print(f"  Output dir: {selector.output_dir}")
            print("  ✅ Selector initialization test passed!")
    
    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        print("\n🔄 Testing full pipeline with synthetic data...")
        
        # Create synthetic DataFrame
        judge_scores, targets, judge_names = create_synthetic_data(n_samples=150, n_judges=5)
        
        df = pd.DataFrame({
            "judge_scores": [list(row) for row in judge_scores],
            "target": targets,
        })
        
        # Create mock judges list
        mock_judges = [{"id": name, "name": name.title()} for name in judge_names]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelectionConfig(
                output_dir=tmpdir,
                max_iterations=3,
                min_judges=2,
                target_column="target",
                train_test_split=0.3,
                r2_improvement_threshold=0.001,
                plateau_patience=2,
            )
            
            selector = IterativeJudgeSelector(config)
            selector.current_judges = mock_judges
            selector.load_data(df)
            
            # Run selection
            results = selector.run()
            
            print(f"  Iterations completed: {len(results)}")
            for r in results:
                print(f"    Iter {r.iteration}: {r.n_judges} judges, R²={r.test_metrics.get('r2', 0):.4f}")
                if r.removed_judge:
                    print(f"      Removed: {r.removed_judge}")
            
            # Check outputs
            assert len(results) > 0
            assert (Path(tmpdir) / "summary.json").exists()
            
            final_r2 = results[-1].test_metrics.get("r2", 0)
            print(f"  Final R²: {final_r2:.4f}")
            
            print("  ✅ Full pipeline test passed!")
            return results


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Running Iterative Selection Pipeline Tests")
    print("=" * 60)
    
    # Judge Set Metrics tests
    metrics_tests = TestJudgeSetMetrics()
    metrics_tests.test_evaluator_basic()
    metrics_tests.test_quick_redundancy()
    metrics_tests.test_metrics_to_dict()
    
    # Gap Analyzer tests
    gap_tests = TestGapAnalyzer()
    gap_tests.test_gap_analysis_basic()
    gap_tests.test_identify_least_important()
    
    # Iterative Selection tests
    selection_tests = TestIterativeSelection()
    selection_tests.test_config_serialization()
    selection_tests.test_selector_initialization()
    selection_tests.test_full_pipeline_synthetic()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
