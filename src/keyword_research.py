"""
YouTube Keyword Research & SEO Optimization

Scrapes YouTube auto-suggest, analyzes trending topics, and generates
long-tail keyword opportunities for better video discoverability.

Features:
- YouTube auto-suggest scraper
- Trending topics tracker
- Long-tail keyword expansion
- Search volume estimation
- Competition analysis
- Keyword clustering

Author: Claude
License: MIT
"""

import requests
import json
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
import time
import re
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeKeywordResearcher:
    """
    YouTube keyword research and SEO optimization tool.
    """

    def __init__(self):
        """Initialize the keyword researcher."""
        self.base_url = "http://suggestqueries.google.com/complete/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Cache for API results
        self.cache = {}
        self.cache_duration = 86400  # 24 hours

    def get_auto_suggestions(self, keyword: str, max_results: int = 10) -> List[str]:
        """
        Get YouTube auto-suggest keywords for a seed keyword.

        Args:
            keyword: Seed keyword to expand
            max_results: Maximum number of suggestions to return

        Returns:
            List of suggested keywords
        """
        # Check cache
        cache_key = f"suggest_{keyword}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                logger.info(f"Using cached suggestions for '{keyword}'")
                return cached_data[:max_results]

        try:
            params = {
                'client': 'youtube',
                'ds': 'yt',
                'q': keyword,
                'hl': 'en'
            }

            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse response - it's JSONP format
            # Response format: window.google.ac.h(["keyword",[["suggestion 1",0],["suggestion 2",0],...]])
            text = response.text

            # Extract JSON array from JSONP
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != 0:
                json_text = text[start:end]
                data = json.loads(json_text)

                # Extract suggestions (they're in nested arrays)
                if len(data) > 1 and isinstance(data[1], list):
                    suggestions = [item[0] for item in data[1] if isinstance(item, list)]

                    # Cache results
                    self.cache[cache_key] = (time.time(), suggestions)

                    logger.info(f"Found {len(suggestions)} suggestions for '{keyword}'")
                    return suggestions[:max_results]

            return []

        except Exception as e:
            logger.error(f"Error getting suggestions for '{keyword}': {e}")
            return []

    def expand_with_modifiers(self, keyword: str, modifiers: Optional[List[str]] = None) -> List[str]:
        """
        Expand a keyword with common modifiers.

        Args:
            keyword: Base keyword
            modifiers: List of modifiers (uses defaults if None)

        Returns:
            List of expanded keywords
        """
        if modifiers is None:
            modifiers = [
                # Questions
                "how to", "what is", "why", "when",
                # Time-based
                "2024", "2025", "new", "latest",
                # Quality
                "best", "top", "good",
                # Duration
                "1 hour", "2 hours", "24/7", "live",
                # Purpose
                "for studying", "for work", "for sleep", "for focus",
                # Mood
                "chill", "relaxing", "calm", "peaceful",
                # Type
                "mix", "playlist", "compilation", "beats",
                # Alphabetical expansion
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
            ]

        expanded = set()

        # Prefix modifiers
        for mod in modifiers:
            expanded.add(f"{mod} {keyword}")
            expanded.add(f"{keyword} {mod}")

        return list(expanded)

    def discover_long_tail_keywords(self, seed_keyword: str, depth: int = 2, max_per_level: int = 5) -> Dict[str, List[str]]:
        """
        Discover long-tail keywords by recursively expanding suggestions.

        Args:
            seed_keyword: Starting keyword
            depth: How many levels deep to search
            max_per_level: Max suggestions to expand at each level

        Returns:
            Dictionary mapping keywords to their suggestions
        """
        logger.info(f"Discovering long-tail keywords for '{seed_keyword}' (depth={depth})")

        results = {}
        to_process = [(seed_keyword, 0)]  # (keyword, current_depth)
        processed = set()

        while to_process:
            current_keyword, current_depth = to_process.pop(0)

            if current_keyword in processed or current_depth >= depth:
                continue

            processed.add(current_keyword)

            # Get suggestions for current keyword
            suggestions = self.get_auto_suggestions(current_keyword, max_results=max_per_level)
            results[current_keyword] = suggestions

            # Add suggestions to processing queue
            if current_depth < depth - 1:
                for suggestion in suggestions[:max_per_level]:
                    if suggestion not in processed:
                        to_process.append((suggestion, current_depth + 1))

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"Discovered {len(results)} keyword clusters")
        return results

    def get_trending_topics(self, category: str = "lofi") -> List[str]:
        """
        Get trending topics in a category.

        Args:
            category: Category to check (e.g., "lofi", "music", "beats")

        Returns:
            List of trending topic keywords
        """
        trending_templates = [
            f"{category} 2025",
            f"{category} trending",
            f"viral {category}",
            f"popular {category}",
            f"new {category}",
            f"{category} compilation",
            f"best {category}",
            f"top {category}"
        ]

        trending = set()

        for template in trending_templates:
            suggestions = self.get_auto_suggestions(template, max_results=10)
            trending.update(suggestions)
            time.sleep(0.3)  # Rate limiting

        return list(trending)

    def analyze_keyword_competition(self, keyword: str) -> Dict[str, any]:
        """
        Analyze competition level for a keyword.

        Args:
            keyword: Keyword to analyze

        Returns:
            Dictionary with competition metrics
        """
        # Get related keywords
        suggestions = self.get_auto_suggestions(keyword, max_results=20)

        # Analyze keyword characteristics
        analysis = {
            'keyword': keyword,
            'word_count': len(keyword.split()),
            'character_count': len(keyword),
            'related_count': len(suggestions),
            'is_long_tail': len(keyword.split()) >= 3,
            'has_numbers': bool(re.search(r'\d', keyword)),
            'has_year': bool(re.search(r'20\d{2}', keyword)),
            'competition_estimate': 'low' if len(keyword.split()) >= 3 else 'high',
            'related_keywords': suggestions[:10],
            'analyzed_at': datetime.now().isoformat()
        }

        # Estimate search intent
        question_words = ['how', 'what', 'why', 'when', 'where', 'who']
        analysis['search_intent'] = 'informational' if any(keyword.lower().startswith(q) for q in question_words) else 'navigational'

        return analysis

    def cluster_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        Cluster keywords by topic/theme.

        Args:
            keywords: List of keywords to cluster

        Returns:
            Dictionary mapping cluster names to keyword lists
        """
        clusters = defaultdict(list)

        # Common themes in lofi music
        themes = {
            'study': ['study', 'focus', 'concentration', 'work', 'homework'],
            'sleep': ['sleep', 'night', 'bedtime', 'insomnia', 'rest'],
            'chill': ['chill', 'relax', 'calm', 'peaceful', 'zen'],
            'mood': ['sad', 'happy', 'cozy', 'rainy', 'sunny'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'weekend'],
            'season': ['winter', 'summer', 'spring', 'fall', 'autumn'],
            'activity': ['coffee', 'reading', 'gaming', 'driving', 'walking'],
            'style': ['jazz', 'hip hop', 'anime', 'japanese', 'korean'],
            'duration': ['1 hour', '2 hours', '3 hours', '24/7', 'live'],
            'quality': ['best', 'top', 'good', 'ultimate', 'perfect']
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()
            matched = False

            for theme, theme_words in themes.items():
                if any(word in keyword_lower for word in theme_words):
                    clusters[theme].append(keyword)
                    matched = True
                    break

            if not matched:
                clusters['general'].append(keyword)

        return dict(clusters)

    def generate_keyword_report(self, seed_keywords: List[str], output_file: str = None) -> Dict[str, any]:
        """
        Generate a comprehensive keyword research report.

        Args:
            seed_keywords: List of seed keywords to research
            output_file: Optional file path to save report

        Returns:
            Complete keyword research report
        """
        logger.info(f"Generating keyword report for {len(seed_keywords)} seeds...")

        report = {
            'generated_at': datetime.now().isoformat(),
            'seed_keywords': seed_keywords,
            'all_keywords': set(),
            'keyword_clusters': {},
            'trending_topics': [],
            'recommendations': []
        }

        # Discover keywords for each seed
        for seed in seed_keywords:
            logger.info(f"Researching seed: {seed}")

            # Get basic suggestions
            suggestions = self.get_auto_suggestions(seed, max_results=15)
            report['all_keywords'].update(suggestions)

            # Expand with modifiers
            expanded = self.expand_with_modifiers(seed)
            for exp_keyword in expanded[:10]:  # Limit to avoid rate limiting
                exp_suggestions = self.get_auto_suggestions(exp_keyword, max_results=5)
                report['all_keywords'].update(exp_suggestions)
                time.sleep(0.3)

            time.sleep(0.5)  # Rate limiting

        # Get trending topics
        report['trending_topics'] = self.get_trending_topics()
        report['all_keywords'].update(report['trending_topics'])

        # Convert set to list
        all_keywords_list = list(report['all_keywords'])

        # Cluster keywords
        report['keyword_clusters'] = self.cluster_keywords(all_keywords_list)

        # Analyze top keywords
        report['analyzed_keywords'] = []
        for keyword in all_keywords_list[:20]:  # Analyze top 20
            analysis = self.analyze_keyword_competition(keyword)
            report['analyzed_keywords'].append(analysis)
            time.sleep(0.3)

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                # Convert set to list for JSON serialization
                report_copy = report.copy()
                report_copy['all_keywords'] = list(report['all_keywords'])
                json.dump(report_copy, f, indent=2)
            logger.info(f"Report saved to {output_file}")

        report['all_keywords'] = all_keywords_list
        return report

    def _generate_recommendations(self, report: Dict) -> List[Dict[str, str]]:
        """Generate keyword recommendations based on analysis."""
        recommendations = []

        # Recommend long-tail keywords
        long_tail = [kw for kw in report['all_keywords'] if len(kw.split()) >= 3]
        if long_tail:
            recommendations.append({
                'type': 'long_tail',
                'priority': 'high',
                'description': 'Use long-tail keywords for easier ranking',
                'examples': long_tail[:5]
            })

        # Recommend trending topics
        if report['trending_topics']:
            recommendations.append({
                'type': 'trending',
                'priority': 'high',
                'description': 'Capitalize on trending topics',
                'examples': report['trending_topics'][:5]
            })

        # Recommend cluster-based content
        for cluster_name, cluster_keywords in report['keyword_clusters'].items():
            if len(cluster_keywords) >= 5:
                recommendations.append({
                    'type': 'cluster',
                    'priority': 'medium',
                    'description': f'Create content around {cluster_name} theme',
                    'examples': cluster_keywords[:3]
                })

        return recommendations


class KeywordOptimizer:
    """
    Optimize video metadata using keyword research.
    """

    def __init__(self, researcher: YouTubeKeywordResearcher = None):
        """Initialize keyword optimizer."""
        self.researcher = researcher or YouTubeKeywordResearcher()

    def optimize_title(self, base_title: str, seed_keywords: List[str]) -> List[str]:
        """
        Generate optimized title variations.

        Args:
            base_title: Base title to optimize
            seed_keywords: Keywords to incorporate

        Returns:
            List of optimized title variations
        """
        variations = []

        # Get suggestions for each seed
        for seed in seed_keywords[:3]:  # Limit to top 3 seeds
            suggestions = self.researcher.get_auto_suggestions(seed, max_results=5)

            # Create title variations
            for suggestion in suggestions:
                # Extract key phrases
                if len(suggestion.split()) > len(seed.split()):
                    # Add trending phrase to title
                    variation = f"{base_title} | {suggestion.title()}"
                    if len(variation) <= 100:  # YouTube title limit
                        variations.append(variation)

        # Add year for freshness
        current_year = datetime.now().year
        variations.append(f"{base_title} ({current_year})")
        variations.append(f"{base_title} - {current_year}")

        return variations[:10]  # Return top 10 variations

    def optimize_description(self, base_description: str, keywords: List[str], max_length: int = 5000) -> str:
        """
        Optimize video description with keywords.

        Args:
            base_description: Base description text
            keywords: Keywords to incorporate
            max_length: Maximum description length

        Returns:
            Optimized description
        """
        # Cluster keywords
        clusters = self.researcher.cluster_keywords(keywords)

        # Build optimized description
        sections = [base_description, "\n"]

        # Add keyword sections
        sections.append("\n🔍 Related Topics:\n")
        for cluster_name, cluster_keywords in list(clusters.items())[:3]:
            if cluster_keywords:
                sections.append(f"\n{cluster_name.title()}: {', '.join(cluster_keywords[:5])}")

        # Add popular searches
        sections.append("\n\n🔎 Popular Searches:\n")
        sections.append(", ".join(keywords[:20]))

        optimized = "".join(sections)

        # Trim if too long
        if len(optimized) > max_length:
            optimized = optimized[:max_length-3] + "..."

        return optimized

    def generate_tags(self, title: str, description: str, seed_keywords: List[str], max_tags: int = 30) -> List[str]:
        """
        Generate optimized tags.

        Args:
            title: Video title
            description: Video description
            seed_keywords: Seed keywords
            max_tags: Maximum number of tags

        Returns:
            List of optimized tags
        """
        tags = set()

        # Add seed keywords
        tags.update(seed_keywords)

        # Get suggestions for each seed
        for seed in seed_keywords[:5]:
            suggestions = self.researcher.get_auto_suggestions(seed, max_results=5)
            tags.update(suggestions)

        # Extract key phrases from title
        title_words = title.lower().split()
        if len(title_words) >= 2:
            # Add 2-3 word combinations
            for i in range(len(title_words) - 1):
                phrase = " ".join(title_words[i:i+2])
                if len(phrase) > 3:  # Avoid very short phrases
                    tags.add(phrase)

        # Sort by length (shorter tags first for better matching)
        sorted_tags = sorted(list(tags), key=len)

        # Ensure character limit (500 chars for YouTube)
        final_tags = []
        total_chars = 0

        for tag in sorted_tags:
            if len(final_tags) >= max_tags:
                break
            if total_chars + len(tag) + 1 <= 500:  # +1 for comma
                final_tags.append(tag)
                total_chars += len(tag) + 1

        return final_tags


# Example usage
if __name__ == "__main__":
    # Initialize researcher
    researcher = YouTubeKeywordResearcher()

    # Test auto-suggest
    print("\n=== Auto-suggest for 'lofi hip hop' ===")
    suggestions = researcher.get_auto_suggestions("lofi hip hop", max_results=10)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

    # Test trending topics
    print("\n=== Trending Topics ===")
    trending = researcher.get_trending_topics("lofi")
    for i, topic in enumerate(trending[:10], 1):
        print(f"{i}. {topic}")

    # Generate keyword report
    print("\n=== Generating Keyword Report ===")
    report = researcher.generate_keyword_report(
        seed_keywords=["lofi hip hop", "study beats", "chill music"],
        output_file="keyword_report.json"
    )

    print(f"\nTotal keywords discovered: {len(report['all_keywords'])}")
    print(f"Keyword clusters: {len(report['keyword_clusters'])}")
    print(f"Recommendations: {len(report['recommendations'])}")

    # Test optimizer
    print("\n=== Title Optimization ===")
    optimizer = KeywordOptimizer(researcher)
    optimized_titles = optimizer.optimize_title(
        "Chill Lofi Beats",
        ["lofi hip hop", "study music"]
    )
    for i, title in enumerate(optimized_titles[:5], 1):
        print(f"{i}. {title}")
