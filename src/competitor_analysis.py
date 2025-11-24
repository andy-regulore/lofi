"""
Competitor Channel Analysis System

Monitors and analyzes competitor YouTube channels to identify successful
strategies, trending content, and growth opportunities.

Features:
- Channel performance tracking
- Video analysis (titles, thumbnails, tags)
- Upload frequency monitoring
- Engagement rate analysis
- Trend detection
- Content gap identification
- Automated email reports

Author: Claude
License: MIT
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoAnalytics:
    """Analytics for a single video."""

    video_id: str
    title: str
    published_at: str
    views: int
    likes: int
    comments: int
    duration: int
    tags: List[str]
    description: str
    thumbnail_url: str

    # Calculated metrics
    engagement_rate: float = 0.0
    views_per_day: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        # Engagement rate: (likes + comments) / views
        if self.views > 0:
            self.engagement_rate = ((self.likes + self.comments) / self.views) * 100

        # Views per day
        pub_date = datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
        days_old = (datetime.now().astimezone() - pub_date).days + 1
        self.views_per_day = self.views / days_old if days_old > 0 else 0


@dataclass
class ChannelAnalytics:
    """Analytics for a YouTube channel."""

    channel_id: str
    channel_name: str
    subscriber_count: int
    total_views: int
    video_count: int
    analyzed_at: str

    # Performance metrics
    avg_views_per_video: float = 0.0
    avg_engagement_rate: float = 0.0
    upload_frequency: float = 0.0  # Videos per week

    # Recent performance
    recent_videos: List[VideoAnalytics] = None
    top_videos: List[VideoAnalytics] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.video_count > 0:
            self.avg_views_per_video = self.total_views / self.video_count


class CompetitorAnalyzer:
    """
    Analyzes competitor YouTube channels.

    Note: This uses YouTube Data API v3. You need an API key.
    Set the API key in config.json under "youtube_api_key"
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the competitor analyzer.

        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://www.googleapis.com/youtube/v3"

        # Tracking data
        self.tracked_channels: Dict[str, ChannelAnalytics] = {}
        self.historical_data: Dict[str, List[ChannelAnalytics]] = defaultdict(list)

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config."""
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                return config.get("youtube_api_key")
        except Exception as e:
            logger.warning(f"Could not load API key from config: {e}")
            return None

    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """
        Get basic channel information.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Channel information dictionary
        """
        if not self.api_key:
            logger.error("No API key configured")
            return None

        import requests

        try:
            url = f"{self.base_url}/channels"
            params = {
                "part": "snippet,statistics,contentDetails",
                "id": channel_id,
                "key": self.api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "items" in data and len(data["items"]) > 0:
                return data["items"][0]
            else:
                logger.warning(f"Channel not found: {channel_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching channel info: {e}")
            return None

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """
        Get recent videos from a channel.

        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch

        Returns:
            List of video information dictionaries
        """
        if not self.api_key:
            logger.error("No API key configured")
            return []

        import requests

        try:
            # First, get the uploads playlist ID
            channel_info = self.get_channel_info(channel_id)
            if not channel_info:
                return []

            uploads_playlist_id = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

            # Get videos from uploads playlist
            url = f"{self.base_url}/playlistItems"
            params = {
                "part": "snippet,contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": min(max_results, 50),
                "key": self.api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            videos = []
            if "items" in data:
                # Get detailed stats for each video
                video_ids = [item["contentDetails"]["videoId"] for item in data["items"]]
                video_stats = self.get_video_stats(video_ids)

                for item, stats in zip(data["items"], video_stats):
                    video_info = {
                        "video_id": item["contentDetails"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["snippet"]["publishedAt"],
                        "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"],
                        **stats,
                    }
                    videos.append(video_info)

            logger.info(f"Fetched {len(videos)} videos from channel {channel_id}")
            return videos

        except Exception as e:
            logger.error(f"Error fetching channel videos: {e}")
            return []

    def get_video_stats(self, video_ids: List[str]) -> List[Dict]:
        """
        Get statistics for multiple videos.

        Args:
            video_ids: List of video IDs

        Returns:
            List of statistics dictionaries
        """
        if not self.api_key or not video_ids:
            return []

        import requests

        try:
            url = f"{self.base_url}/videos"
            params = {
                "part": "statistics,contentDetails,snippet",
                "id": ",".join(video_ids[:50]),  # API limit
                "key": self.api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            stats_list = []
            if "items" in data:
                for item in data["items"]:
                    stats = item["statistics"]
                    content = item["contentDetails"]
                    snippet = item["snippet"]

                    # Parse duration (ISO 8601 format)
                    duration_str = content.get("duration", "PT0S")
                    duration_seconds = self._parse_duration(duration_str)

                    stats_dict = {
                        "views": int(stats.get("viewCount", 0)),
                        "likes": int(stats.get("likeCount", 0)),
                        "comments": int(stats.get("commentCount", 0)),
                        "duration": duration_seconds,
                        "tags": snippet.get("tags", []),
                    }
                    stats_list.append(stats_dict)

            return stats_list

        except Exception as e:
            logger.error(f"Error fetching video stats: {e}")
            return [{"views": 0, "likes": 0, "comments": 0, "duration": 0, "tags": []}] * len(
                video_ids
            )

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        import re

        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match = re.match(pattern, duration_str)

        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds

        return 0

    def analyze_channel(self, channel_id: str, video_limit: int = 50) -> Optional[ChannelAnalytics]:
        """
        Perform comprehensive analysis of a competitor channel.

        Args:
            channel_id: YouTube channel ID
            video_limit: Number of recent videos to analyze

        Returns:
            ChannelAnalytics object with comprehensive metrics
        """
        logger.info(f"Analyzing channel: {channel_id}")

        # Get channel information
        channel_info = self.get_channel_info(channel_id)
        if not channel_info:
            return None

        snippet = channel_info["snippet"]
        stats = channel_info["statistics"]

        # Get recent videos
        videos = self.get_channel_videos(channel_id, max_results=video_limit)

        # Create VideoAnalytics objects
        video_analytics = []
        for video in videos:
            va = VideoAnalytics(
                video_id=video["video_id"],
                title=video["title"],
                published_at=video["published_at"],
                views=video["views"],
                likes=video["likes"],
                comments=video["comments"],
                duration=video["duration"],
                tags=video["tags"],
                description=video["description"],
                thumbnail_url=video["thumbnail_url"],
            )
            video_analytics.append(va)

        # Calculate aggregate metrics
        if video_analytics:
            avg_engagement = sum(v.engagement_rate for v in video_analytics) / len(video_analytics)

            # Calculate upload frequency
            if len(video_analytics) >= 2:
                first_date = datetime.fromisoformat(
                    video_analytics[-1].published_at.replace("Z", "+00:00")
                )
                last_date = datetime.fromisoformat(
                    video_analytics[0].published_at.replace("Z", "+00:00")
                )
                weeks = max((last_date - first_date).days / 7, 1)
                upload_frequency = len(video_analytics) / weeks
            else:
                upload_frequency = 0
        else:
            avg_engagement = 0
            upload_frequency = 0

        # Sort videos by performance
        top_videos = sorted(video_analytics, key=lambda v: v.views, reverse=True)[:10]

        # Create channel analytics
        analytics = ChannelAnalytics(
            channel_id=channel_id,
            channel_name=snippet["title"],
            subscriber_count=int(stats.get("subscriberCount", 0)),
            total_views=int(stats.get("viewCount", 0)),
            video_count=int(stats.get("videoCount", 0)),
            analyzed_at=datetime.now().isoformat(),
            avg_engagement_rate=avg_engagement,
            upload_frequency=upload_frequency,
            recent_videos=video_analytics[:20],
            top_videos=top_videos,
        )

        # Calculate average views per video
        if analytics.video_count > 0:
            analytics.avg_views_per_video = analytics.total_views / analytics.video_count

        # Store analytics
        self.tracked_channels[channel_id] = analytics
        self.historical_data[channel_id].append(analytics)

        logger.info(f"Analysis complete for {analytics.channel_name}")
        return analytics

    def compare_channels(self, channel_ids: List[str]) -> Dict:
        """
        Compare multiple competitor channels.

        Args:
            channel_ids: List of channel IDs to compare

        Returns:
            Comparison report
        """
        logger.info(f"Comparing {len(channel_ids)} channels...")

        # Analyze each channel
        analyses = []
        for channel_id in channel_ids:
            analytics = self.analyze_channel(channel_id)
            if analytics:
                analyses.append(analytics)
            time.sleep(1)  # Rate limiting

        if not analyses:
            return {}

        # Create comparison report
        report = {
            "analyzed_at": datetime.now().isoformat(),
            "channels": [asdict(a) for a in analyses],
            "rankings": {
                "by_subscribers": sorted(analyses, key=lambda a: a.subscriber_count, reverse=True),
                "by_avg_views": sorted(analyses, key=lambda a: a.avg_views_per_video, reverse=True),
                "by_engagement": sorted(
                    analyses, key=lambda a: a.avg_engagement_rate, reverse=True
                ),
                "by_upload_frequency": sorted(
                    analyses, key=lambda a: a.upload_frequency, reverse=True
                ),
            },
            "insights": self._generate_insights(analyses),
        }

        # Convert rankings to simple format
        for rank_type in report["rankings"]:
            report["rankings"][rank_type] = [
                {"channel": a.channel_name, "value": getattr(a, rank_type.replace("by_", ""))}
                for a in report["rankings"][rank_type]
            ]

        return report

    def _generate_insights(self, analyses: List[ChannelAnalytics]) -> List[Dict]:
        """Generate insights from channel comparisons."""
        insights = []

        if not analyses:
            return insights

        # Average metrics
        avg_subs = sum(a.subscriber_count for a in analyses) / len(analyses)
        avg_views = sum(a.avg_views_per_video for a in analyses) / len(analyses)
        avg_engagement = sum(a.avg_engagement_rate for a in analyses) / len(analyses)
        avg_frequency = sum(a.upload_frequency for a in analyses) / len(analyses)

        insights.append(
            {
                "type": "benchmark",
                "description": "Industry averages",
                "metrics": {
                    "avg_subscribers": int(avg_subs),
                    "avg_views_per_video": int(avg_views),
                    "avg_engagement_rate": round(avg_engagement, 2),
                    "avg_upload_frequency": round(avg_frequency, 2),
                },
            }
        )

        # Find top performer
        top_channel = max(analyses, key=lambda a: a.subscriber_count)
        insights.append(
            {
                "type": "top_performer",
                "description": f"{top_channel.channel_name} is the top channel by subscribers",
                "metrics": {
                    "subscribers": top_channel.subscriber_count,
                    "avg_views": int(top_channel.avg_views_per_video),
                    "engagement_rate": round(top_channel.avg_engagement_rate, 2),
                },
            }
        )

        # Find best engagement
        best_engagement = max(analyses, key=lambda a: a.avg_engagement_rate)
        if best_engagement != top_channel:
            insights.append(
                {
                    "type": "high_engagement",
                    "description": f"{best_engagement.channel_name} has the highest engagement rate",
                    "metrics": {
                        "engagement_rate": round(best_engagement.avg_engagement_rate, 2),
                        "strategy": "Study their content format and community interaction",
                    },
                }
            )

        # Upload frequency insights
        most_active = max(analyses, key=lambda a: a.upload_frequency)
        insights.append(
            {
                "type": "upload_frequency",
                "description": f"{most_active.channel_name} uploads most frequently",
                "metrics": {
                    "videos_per_week": round(most_active.upload_frequency, 2),
                    "recommendation": "Consider increasing upload frequency to match",
                },
            }
        )

        return insights

    def identify_trending_topics(self, channel_ids: List[str]) -> List[Dict]:
        """
        Identify trending topics across competitor channels.

        Args:
            channel_ids: List of channels to analyze

        Returns:
            List of trending topics with metrics
        """
        logger.info("Identifying trending topics...")

        all_titles = []
        all_tags = []

        for channel_id in channel_ids:
            videos = self.get_channel_videos(channel_id, max_results=20)
            for video in videos:
                all_titles.append(video["title"].lower())
                all_tags.extend([tag.lower() for tag in video["tags"]])
            time.sleep(0.5)

        # Count word frequency in titles
        import re
        from collections import Counter

        # Extract words from titles
        words = []
        for title in all_titles:
            # Remove common words
            title_words = re.findall(r"\b\w{4,}\b", title)  # Words with 4+ chars
            words.extend(title_words)

        # Count frequencies
        word_freq = Counter(words)
        tag_freq = Counter(all_tags)

        # Identify trends
        trending = []

        # Top keywords in titles
        for word, count in word_freq.most_common(20):
            if count >= 3:  # Appears in at least 3 videos
                trending.append(
                    {
                        "keyword": word,
                        "frequency": count,
                        "source": "title",
                        "opportunity": "high" if count >= 5 else "medium",
                    }
                )

        # Top tags
        for tag, count in tag_freq.most_common(20):
            if count >= 3:
                trending.append(
                    {
                        "keyword": tag,
                        "frequency": count,
                        "source": "tag",
                        "opportunity": "high" if count >= 5 else "medium",
                    }
                )

        # Sort by frequency
        trending.sort(key=lambda x: x["frequency"], reverse=True)

        logger.info(f"Identified {len(trending)} trending topics")
        return trending

    def generate_report(
        self, channel_ids: List[str], output_file: str = "competitor_report.json"
    ) -> Dict:
        """
        Generate a comprehensive competitor analysis report.

        Args:
            channel_ids: List of competitor channel IDs
            output_file: File path to save report

        Returns:
            Complete analysis report
        """
        logger.info(f"Generating competitor report for {len(channel_ids)} channels...")

        # Compare channels
        comparison = self.compare_channels(channel_ids)

        # Identify trends
        trending = self.identify_trending_topics(channel_ids)

        # Compile report
        report = {
            "generated_at": datetime.now().isoformat(),
            "channel_comparison": comparison,
            "trending_topics": trending,
            "recommendations": self._generate_recommendations(comparison, trending),
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {output_file}")
        return report

    def _generate_recommendations(self, comparison: Dict, trending: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []

        if comparison and "insights" in comparison:
            benchmarks = next((i for i in comparison["insights"] if i["type"] == "benchmark"), None)

            if benchmarks:
                recommendations.append(
                    {
                        "category": "upload_frequency",
                        "priority": "high",
                        "action": f"Upload at least {benchmarks['metrics']['avg_upload_frequency']:.1f} videos per week",
                        "reason": "Match industry average",
                    }
                )

                recommendations.append(
                    {
                        "category": "engagement",
                        "priority": "high",
                        "action": f"Target {benchmarks['metrics']['avg_engagement_rate']:.1f}% engagement rate",
                        "reason": "Industry benchmark",
                    }
                )

        # Trending topic recommendations
        if trending:
            top_trends = [t["keyword"] for t in trending[:5]]
            recommendations.append(
                {
                    "category": "content",
                    "priority": "high",
                    "action": f"Create content around: {', '.join(top_trends)}",
                    "reason": "Currently trending across competitors",
                }
            )

        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize analyzer (requires YouTube API key)
    analyzer = CompetitorAnalyzer(api_key="YOUR_API_KEY_HERE")

    # Example competitor channel IDs (LoFi channels)
    competitors = [
        "UCSJ4gkVC6NrvII8umztf0Ow",  # Lofi Girl
        "UCOxqgCwgOqC2lMqC5PYz_Dg",  # ChilledCow (old)
        # Add more channel IDs
    ]

    # Generate comprehensive report
    report = analyzer.generate_report(competitors)

    print("\n=== Competitor Analysis Report ===")
    print(f"Analyzed {len(competitors)} channels")
    print(f"Identified {len(report['trending_topics'])} trending topics")
    print(f"Generated {len(report['recommendations'])} recommendations")
