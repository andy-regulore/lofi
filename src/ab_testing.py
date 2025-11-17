"""
A/B Testing Automation System

Generates multiple variations of content (titles, thumbnails, descriptions)
and automatically publishes the best-performing versions based on analytics.

Features:
- Generate 2-3 variations per track
- Track performance metrics
- Statistical significance testing
- Auto-publish winners
- Experiment management
- Performance reporting

Author: Claude
License: MIT
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import random
from collections import defaultdict
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Variation:
    """A single variation in an A/B test."""
    variation_id: str
    variation_name: str  # "A", "B", "C"

    # Content variations
    title: str
    description: str
    tags: List[str]
    thumbnail_path: Optional[str] = None

    # Performance metrics
    impressions: int = 0
    views: int = 0
    clicks: int = 0
    watch_time: float = 0.0
    likes: int = 0
    comments: int = 0
    shares: int = 0

    # Calculated metrics
    ctr: float = 0.0  # Click-through rate
    engagement_rate: float = 0.0
    avg_view_duration: float = 0.0

    # Status
    is_published: bool = False
    published_at: Optional[str] = None
    video_id: Optional[str] = None

    def update_metrics(self, metrics: Dict):
        """Update performance metrics and calculate derived values."""
        self.impressions = metrics.get('impressions', self.impressions)
        self.views = metrics.get('views', self.views)
        self.clicks = metrics.get('clicks', self.clicks)
        self.watch_time = metrics.get('watch_time', self.watch_time)
        self.likes = metrics.get('likes', self.likes)
        self.comments = metrics.get('comments', self.comments)
        self.shares = metrics.get('shares', self.shares)

        # Calculate CTR
        if self.impressions > 0:
            self.ctr = (self.clicks / self.impressions) * 100

        # Calculate engagement rate
        if self.views > 0:
            total_engagement = self.likes + self.comments + self.shares
            self.engagement_rate = (total_engagement / self.views) * 100

            # Average view duration
            if self.watch_time > 0:
                self.avg_view_duration = self.watch_time / self.views


@dataclass
class ABTest:
    """An A/B test experiment."""
    test_id: str
    test_name: str
    created_at: str

    # Test configuration
    variations: List[Variation]
    test_duration_days: int = 7
    min_sample_size: int = 100  # Minimum views per variation
    confidence_level: float = 0.95  # 95% confidence

    # Test status
    status: str = "draft"  # draft, running, completed, winner_selected
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    winner_id: Optional[str] = None

    # Results
    results: Optional[Dict] = None

    def start_test(self):
        """Start the A/B test."""
        self.status = "running"
        self.started_at = datetime.now().isoformat()
        logger.info(f"Started A/B test: {self.test_name}")

    def end_test(self):
        """End the A/B test."""
        self.status = "completed"
        self.ended_at = datetime.now().isoformat()
        logger.info(f"Ended A/B test: {self.test_name}")

    def has_sufficient_data(self) -> bool:
        """Check if test has sufficient data for analysis."""
        return all(v.views >= self.min_sample_size for v in self.variations)

    def has_reached_duration(self) -> bool:
        """Check if test has run for sufficient time."""
        if not self.started_at:
            return False

        start_time = datetime.fromisoformat(self.started_at)
        elapsed = datetime.now() - start_time
        return elapsed.days >= self.test_duration_days


class ABTestManager:
    """
    Manages A/B testing experiments for content optimization.
    """

    def __init__(self, storage_path: str = "data/ab_tests"):
        """
        Initialize A/B test manager.

        Args:
            storage_path: Directory to store test data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, ABTest] = {}

        # Load existing tests
        self._load_tests()

    def create_variations(self, base_content: Dict, num_variations: int = 3) -> List[Variation]:
        """
        Create multiple variations of content.

        Args:
            base_content: Base content with title, description, tags
            num_variations: Number of variations to create (2-3)

        Returns:
            List of Variation objects
        """
        variations = []
        variation_names = ['A', 'B', 'C', 'D', 'E'][:num_variations]

        base_title = base_content.get('title', '')
        base_description = base_content.get('description', '')
        base_tags = base_content.get('tags', [])

        for i, name in enumerate(variation_names):
            # Create title variations
            if i == 0:
                # Variation A: Original
                title = base_title
            elif i == 1:
                # Variation B: Add year/timestamp
                title = f"{base_title} ({datetime.now().year})"
            elif i == 2:
                # Variation C: Add emoji/descriptive prefix
                prefixes = ["🎵 ", "✨ ", "🌙 ", "☕ ", "📚 "]
                title = f"{random.choice(prefixes)}{base_title}"
            else:
                # Additional variations: Rearrange or add modifiers
                modifiers = ["Best", "Ultimate", "Perfect", "Relaxing", "Chill"]
                title = f"{random.choice(modifiers)} {base_title}"

            # Create description variations
            if i == 0:
                description = base_description
            elif i == 1:
                # Add call-to-action at the beginning
                description = "👉 Subscribe for more! 👈\n\n" + base_description
            else:
                # Add social proof
                description = "Join thousands of listeners! 🎧\n\n" + base_description

            # Tag variations (slight differences)
            tags = base_tags.copy()
            if i == 1:
                # Add trending tags
                tags.extend(["trending", "viral", "popular"])
            elif i == 2:
                # Add long-tail tags
                tags.extend(["study music", "work music", "focus beats"])

            # Remove duplicates and limit
            tags = list(dict.fromkeys(tags))[:30]

            variation = Variation(
                variation_id=f"var_{i}_{int(time.time())}",
                variation_name=name,
                title=title,
                description=description,
                tags=tags
            )
            variations.append(variation)

        logger.info(f"Created {len(variations)} variations")
        return variations

    def create_test(self, test_name: str, base_content: Dict,
                   num_variations: int = 3, test_duration_days: int = 7) -> ABTest:
        """
        Create a new A/B test.

        Args:
            test_name: Name for the test
            base_content: Base content to create variations from
            num_variations: Number of variations (2-3)
            test_duration_days: Test duration in days

        Returns:
            ABTest object
        """
        # Create variations
        variations = self.create_variations(base_content, num_variations)

        # Create test
        test = ABTest(
            test_id=f"test_{int(time.time())}",
            test_name=test_name,
            created_at=datetime.now().isoformat(),
            variations=variations,
            test_duration_days=test_duration_days
        )

        # Store test
        self.active_tests[test.test_id] = test
        self._save_test(test)

        logger.info(f"Created A/B test: {test_name} with {len(variations)} variations")
        return test

    def update_variation_metrics(self, test_id: str, variation_id: str, metrics: Dict):
        """
        Update metrics for a variation.

        Args:
            test_id: Test ID
            variation_id: Variation ID
            metrics: Dictionary of metrics
        """
        if test_id not in self.active_tests:
            logger.warning(f"Test {test_id} not found")
            return

        test = self.active_tests[test_id]

        # Find and update variation
        for variation in test.variations:
            if variation.variation_id == variation_id:
                variation.update_metrics(metrics)
                logger.info(f"Updated metrics for {variation.variation_name}: "
                          f"{variation.views} views, {variation.ctr:.2f}% CTR")
                break

        # Save updated test
        self._save_test(test)

    def analyze_test(self, test_id: str) -> Dict:
        """
        Analyze A/B test results with statistical significance.

        Args:
            test_id: Test ID

        Returns:
            Analysis results
        """
        if test_id not in self.active_tests:
            logger.warning(f"Test {test_id} not found")
            return {}

        test = self.active_tests[test_id]

        if not test.has_sufficient_data():
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {test.min_sample_size} views per variation',
                'current_data': {v.variation_name: v.views for v in test.variations}
            }

        # Calculate metrics for each variation
        results = {
            'test_id': test_id,
            'test_name': test.test_name,
            'analyzed_at': datetime.now().isoformat(),
            'variations': []
        }

        for variation in test.variations:
            var_result = {
                'name': variation.variation_name,
                'title': variation.title,
                'metrics': {
                    'views': variation.views,
                    'ctr': round(variation.ctr, 2),
                    'engagement_rate': round(variation.engagement_rate, 2),
                    'avg_view_duration': round(variation.avg_view_duration, 2),
                    'likes': variation.likes,
                    'comments': variation.comments
                }
            }
            results['variations'].append(var_result)

        # Determine winner based on primary metric (views or CTR)
        # Sort by engagement rate (combination of multiple factors)
        ranked = sorted(test.variations,
                       key=lambda v: (v.engagement_rate, v.ctr, v.views),
                       reverse=True)

        winner = ranked[0]
        results['winner'] = {
            'variation_name': winner.variation_name,
            'title': winner.title,
            'improvement_vs_baseline': self._calculate_improvement(test.variations[0], winner)
        }

        # Statistical significance test (simplified chi-square)
        results['statistical_significance'] = self._test_significance(test.variations)

        # Save results
        test.results = results
        test.winner_id = winner.variation_id

        logger.info(f"Analysis complete. Winner: {winner.variation_name}")
        return results

    def _calculate_improvement(self, baseline: Variation, winner: Variation) -> Dict:
        """Calculate improvement percentage vs baseline."""
        improvements = {}

        if baseline.views > 0:
            improvements['views'] = round(((winner.views - baseline.views) / baseline.views) * 100, 2)

        if baseline.ctr > 0:
            improvements['ctr'] = round(((winner.ctr - baseline.ctr) / baseline.ctr) * 100, 2)

        if baseline.engagement_rate > 0:
            improvements['engagement'] = round(
                ((winner.engagement_rate - baseline.engagement_rate) / baseline.engagement_rate) * 100, 2)

        return improvements

    def _test_significance(self, variations: List[Variation]) -> Dict:
        """
        Simplified statistical significance test.

        Uses chi-square test for CTR comparison.
        """
        if len(variations) < 2:
            return {'is_significant': False, 'reason': 'Need at least 2 variations'}

        # Calculate chi-square for CTR
        total_impressions = sum(v.impressions for v in variations)
        total_clicks = sum(v.clicks for v in variations)

        if total_impressions == 0:
            return {'is_significant': False, 'reason': 'No impression data'}

        expected_ctr = total_clicks / total_impressions if total_impressions > 0 else 0

        chi_square = 0
        for variation in variations:
            if variation.impressions > 0:
                expected_clicks = variation.impressions * expected_ctr
                if expected_clicks > 0:
                    chi_square += ((variation.clicks - expected_clicks) ** 2) / expected_clicks

        # Degrees of freedom
        df = len(variations) - 1

        # Critical values for 95% confidence (simplified)
        critical_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
        critical_value = critical_values.get(df, 9.488)

        is_significant = chi_square > critical_value

        return {
            'is_significant': is_significant,
            'chi_square': round(chi_square, 3),
            'critical_value': critical_value,
            'confidence_level': '95%',
            'interpretation': 'Results are statistically significant' if is_significant
                            else 'Results are not statistically significant'
        }

    def auto_select_winner(self, test_id: str) -> Optional[str]:
        """
        Automatically select and publish the winning variation.

        Args:
            test_id: Test ID

        Returns:
            Winner variation ID or None
        """
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]

        # Check if test is ready
        if not test.has_sufficient_data() or not test.has_reached_duration():
            logger.info(f"Test {test_id} not ready for winner selection")
            return None

        # Analyze test
        results = self.analyze_test(test_id)

        # Check statistical significance
        if not results.get('statistical_significance', {}).get('is_significant', False):
            logger.warning(f"Test {test_id} results not statistically significant")
            # Still select winner but with warning

        # Mark test as completed
        test.end_test()
        test.status = "winner_selected"

        # Move to completed tests
        self.completed_tests[test_id] = test
        del self.active_tests[test_id]

        self._save_test(test)

        winner_name = results['winner']['variation_name']
        logger.info(f"Auto-selected winner: Variation {winner_name}")

        return test.winner_id

    def get_winner_content(self, test_id: str) -> Optional[Dict]:
        """
        Get the winning variation's content for publishing.

        Args:
            test_id: Test ID

        Returns:
            Winner content dictionary
        """
        test = self.completed_tests.get(test_id) or self.active_tests.get(test_id)

        if not test or not test.winner_id:
            return None

        # Find winner variation
        winner = next((v for v in test.variations if v.variation_id == test.winner_id), None)

        if not winner:
            return None

        return {
            'title': winner.title,
            'description': winner.description,
            'tags': winner.tags,
            'thumbnail_path': winner.thumbnail_path,
            'variation_name': winner.variation_name,
            'performance': {
                'views': winner.views,
                'ctr': winner.ctr,
                'engagement_rate': winner.engagement_rate
            }
        }

    def generate_report(self, test_id: str, output_file: Optional[str] = None) -> Dict:
        """
        Generate comprehensive A/B test report.

        Args:
            test_id: Test ID
            output_file: Optional file path to save report

        Returns:
            Complete test report
        """
        test = self.completed_tests.get(test_id) or self.active_tests.get(test_id)

        if not test:
            return {}

        # Analyze if not already done
        if not test.results:
            test.results = self.analyze_test(test_id)

        report = {
            'test_summary': {
                'test_id': test.test_id,
                'test_name': test.test_name,
                'status': test.status,
                'created_at': test.created_at,
                'started_at': test.started_at,
                'ended_at': test.ended_at,
                'duration_days': test.test_duration_days
            },
            'variations': [asdict(v) for v in test.variations],
            'results': test.results,
            'winner': self.get_winner_content(test_id) if test.winner_id else None,
            'recommendations': self._generate_recommendations(test)
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")

        return report

    def _generate_recommendations(self, test: ABTest) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if not test.results:
            return recommendations

        winner_data = test.results.get('winner', {})
        improvements = winner_data.get('improvement_vs_baseline', {})

        # Title recommendations
        winner_var = next((v for v in test.variations if v.variation_id == test.winner_id), None)
        if winner_var:
            if "(" in winner_var.title and ")" in winner_var.title:
                recommendations.append("Adding year/date to titles increases performance")
            if any(emoji in winner_var.title for emoji in ["🎵", "✨", "🌙", "☕", "📚"]):
                recommendations.append("Emoji in titles improve click-through rate")

        # Engagement recommendations
        if improvements.get('engagement', 0) > 10:
            recommendations.append(f"Winner variation improved engagement by {improvements['engagement']}%")

        # CTR recommendations
        if improvements.get('ctr', 0) > 15:
            recommendations.append(f"Winning title format increased CTR by {improvements['ctr']}%")

        # Statistical significance
        if test.results.get('statistical_significance', {}).get('is_significant'):
            recommendations.append("Results are statistically significant - confidently use winning variation")
        else:
            recommendations.append("Results not statistically significant - consider running longer test")

        return recommendations

    def _save_test(self, test: ABTest):
        """Save test to storage."""
        file_path = self.storage_path / f"{test.test_id}.json"
        with open(file_path, 'w') as f:
            json.dump(asdict(test), f, indent=2, default=str)

    def _load_tests(self):
        """Load tests from storage."""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # Reconstruct variations
                    variations = [Variation(**v) for v in data['variations']]
                    data['variations'] = variations

                    # Create ABTest object
                    test = ABTest(**data)

                    # Store in appropriate dict
                    if test.status in ['completed', 'winner_selected']:
                        self.completed_tests[test.test_id] = test
                    else:
                        self.active_tests[test.test_id] = test

            except Exception as e:
                logger.error(f"Error loading test from {file_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ABTestManager()

    # Create base content
    base_content = {
        'title': "Chill Lofi Beats for Studying",
        'description': "Relaxing lofi hip hop music perfect for studying, working, or just chilling.",
        'tags': ["lofi", "study music", "chill beats", "hip hop", "relaxing music"]
    }

    # Create A/B test
    test = manager.create_test(
        test_name="Title Format Test - Study Beats",
        base_content=base_content,
        num_variations=3,
        test_duration_days=7
    )

    print(f"\n=== Created A/B Test: {test.test_name} ===")
    for var in test.variations:
        print(f"\nVariation {var.variation_name}:")
        print(f"  Title: {var.title}")

    # Start test
    test.start_test()

    # Simulate metric updates (in real usage, these come from YouTube Analytics)
    print("\n=== Simulating Performance Data ===")
    manager.update_variation_metrics(test.test_id, test.variations[0].variation_id, {
        'impressions': 1000, 'clicks': 50, 'views': 45, 'watch_time': 1800, 'likes': 5, 'comments': 2
    })
    manager.update_variation_metrics(test.test_id, test.variations[1].variation_id, {
        'impressions': 1000, 'clicks': 65, 'views': 60, 'watch_time': 2400, 'likes': 8, 'comments': 3
    })
    manager.update_variation_metrics(test.test_id, test.variations[2].variation_id, {
        'impressions': 1000, 'clicks': 55, 'views': 50, 'watch_time': 2000, 'likes': 6, 'comments': 2
    })

    # Analyze results
    print("\n=== Analyzing Test Results ===")
    results = manager.analyze_test(test.test_id)

    if results:
        print(f"\nWinner: Variation {results['winner']['variation_name']}")
        print(f"Title: {results['winner']['title']}")
        print(f"\nImprovements vs Baseline:")
        for metric, value in results['winner']['improvement_vs_baseline'].items():
            print(f"  {metric}: {value:+.1f}%")

        print(f"\nStatistical Significance: {results['statistical_significance']['interpretation']}")

    # Generate report
    report = manager.generate_report(test.test_id, "ab_test_report.json")
    print(f"\n=== Recommendations ===")
    for rec in report['recommendations']:
        print(f"  • {rec}")
