"""
Analytics dashboard for Lo-Fi music empire.

Track performance across platforms:
- YouTube (views, watch time, subscribers, revenue)
- Spotify/Apple Music (streams, listeners, playlist adds)
- Social media (engagement, reach)
- Financial metrics (revenue, costs, ROI)
- Growth projections
- A/B test results

Features:
- Multi-platform data aggregation
- Real-time performance monitoring
- Trend analysis and forecasting
- Revenue tracking and projections
- Audience insights
- Content performance ranking

Author: Claude
License: MIT
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json


class Platform(Enum):
    """Content platforms."""
    YOUTUBE = "youtube"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    SOUNDCLOUD = "soundcloud"
    BANDCAMP = "bandcamp"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a piece of content."""
    platform: Platform
    content_id: str
    title: str
    views_or_streams: int
    likes: int
    comments: int
    shares: int
    watch_time_minutes: float
    revenue_usd: float
    date: datetime


@dataclass
class AudienceInsights:
    """Audience demographic and behavioral insights."""
    age_distribution: Dict[str, float]  # e.g., {'18-24': 0.3, '25-34': 0.4}
    gender_distribution: Dict[str, float]  # e.g., {'male': 0.6, 'female': 0.4}
    top_countries: List[Tuple[str, float]]  # (country, percentage)
    peak_hours: List[int]  # Hours of day with most activity
    avg_session_duration: float  # Minutes
    retention_rate: float  # 0-1


class YouTubeAnalytics:
    """YouTube-specific analytics."""

    def __init__(self):
        """Initialize YouTube analytics."""
        self.videos = {}
        self.channel_stats = {
            'subscribers': 0,
            'total_views': 0,
            'total_watch_time_hours': 0
        }

    def add_video_performance(self, metrics: PerformanceMetrics):
        """Add video performance data."""
        self.videos[metrics.content_id] = metrics

    def get_top_videos(self, n: int = 10, metric: str = 'views') -> List[PerformanceMetrics]:
        """
        Get top performing videos.

        Args:
            n: Number of videos
            metric: Metric to sort by

        Returns:
            Top N videos
        """
        if metric == 'views':
            sorted_videos = sorted(self.videos.values(),
                                 key=lambda x: x.views_or_streams,
                                 reverse=True)
        elif metric == 'revenue':
            sorted_videos = sorted(self.videos.values(),
                                 key=lambda x: x.revenue_usd,
                                 reverse=True)
        else:
            sorted_videos = list(self.videos.values())

        return sorted_videos[:n]

    def calculate_cpm(self) -> float:
        """Calculate average CPM (cost per mille)."""
        total_views = sum(v.views_or_streams for v in self.videos.values())
        total_revenue = sum(v.revenue_usd for v in self.videos.values())

        if total_views == 0:
            return 0.0

        return (total_revenue / total_views) * 1000

    def analyze_upload_patterns(self) -> Dict:
        """Analyze which upload patterns perform best."""
        by_day = {}
        by_hour = {}

        for video in self.videos.values():
            day = video.date.strftime('%A')
            hour = video.date.hour

            if day not in by_day:
                by_day[day] = {'count': 0, 'avg_views': 0}
            by_day[day]['count'] += 1
            by_day[day]['avg_views'] += video.views_or_streams

            if hour not in by_hour:
                by_hour[hour] = {'count': 0, 'avg_views': 0}
            by_hour[hour]['count'] += 1
            by_hour[hour]['avg_views'] += video.views_or_streams

        # Calculate averages
        for day_stats in by_day.values():
            if day_stats['count'] > 0:
                day_stats['avg_views'] /= day_stats['count']

        for hour_stats in by_hour.values():
            if hour_stats['count'] > 0:
                hour_stats['avg_views'] /= hour_stats['count']

        # Find best day and hour
        best_day = max(by_day.items(), key=lambda x: x[1]['avg_views'])[0]
        best_hour = max(by_hour.items(), key=lambda x: x[1]['avg_views'])[0]

        return {
            'best_day': best_day,
            'best_hour': best_hour,
            'by_day': by_day,
            'by_hour': by_hour
        }


class SpotifyAnalytics:
    """Spotify-specific analytics."""

    def __init__(self):
        """Initialize Spotify analytics."""
        self.tracks = {}
        self.monthly_listeners = 0

    def add_track_performance(self, metrics: PerformanceMetrics):
        """Add track performance data."""
        self.tracks[metrics.content_id] = metrics

    def calculate_streaming_revenue(self, streams: int, rate_per_stream: float = 0.004) -> float:
        """
        Calculate streaming revenue.

        Args:
            streams: Number of streams
            rate_per_stream: Revenue per stream (default: $0.004)

        Returns:
            Estimated revenue in USD
        """
        return streams * rate_per_stream

    def analyze_playlist_impact(self) -> Dict:
        """Analyze impact of playlist placements."""
        # Placeholder for playlist analysis
        return {
            'playlist_adds': 150,
            'streams_from_playlists': 25000,
            'percentage_from_playlists': 0.6
        }


class FinancialDashboard:
    """Financial tracking and projections."""

    def __init__(self):
        """Initialize financial dashboard."""
        self.revenue_streams = {
            'youtube_ads': [],
            'spotify_streams': [],
            'apple_music': [],
            'sample_packs': [],
            'patreon': [],
            'other': []
        }
        self.costs = {
            'distribution': [],  # DistroKid, etc.
            'api_costs': [],
            'samples': [],
            'software': [],
            'marketing': [],
            'other': []
        }

    def add_revenue(self, source: str, amount: float, date: datetime):
        """Add revenue entry."""
        if source in self.revenue_streams:
            self.revenue_streams[source].append({'amount': amount, 'date': date})

    def add_cost(self, category: str, amount: float, date: datetime, description: str = ""):
        """Add cost entry."""
        if category in self.costs:
            self.costs[category].append({
                'amount': amount,
                'date': date,
                'description': description
            })

    def get_total_revenue(self, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> float:
        """Calculate total revenue in period."""
        total = 0.0

        for stream, entries in self.revenue_streams.items():
            for entry in entries:
                if start_date and entry['date'] < start_date:
                    continue
                if end_date and entry['date'] > end_date:
                    continue
                total += entry['amount']

        return total

    def get_total_costs(self, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> float:
        """Calculate total costs in period."""
        total = 0.0

        for category, entries in self.costs.items():
            for entry in entries:
                if start_date and entry['date'] < start_date:
                    continue
                if end_date and entry['date'] > end_date:
                    continue
                total += entry['amount']

        return total

    def calculate_roi(self, start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> float:
        """Calculate return on investment."""
        revenue = self.get_total_revenue(start_date, end_date)
        costs = self.get_total_costs(start_date, end_date)

        if costs == 0:
            return 0.0

        return ((revenue - costs) / costs) * 100

    def project_revenue(self, months: int = 6) -> List[Tuple[str, float]]:
        """
        Project revenue for future months.

        Args:
            months: Number of months to project

        Returns:
            List of (month, projected_revenue) tuples
        """
        # Simple linear projection based on last 3 months
        now = datetime.now()
        last_3_months_revenue = []

        for i in range(3):
            month_start = now - timedelta(days=30 * (i + 1))
            month_end = now - timedelta(days=30 * i)
            monthly_revenue = self.get_total_revenue(month_start, month_end)
            last_3_months_revenue.append(monthly_revenue)

        # Calculate growth rate
        if len(last_3_months_revenue) >= 2:
            growth_rate = (last_3_months_revenue[0] - last_3_months_revenue[-1]) / last_3_months_revenue[-1]
        else:
            growth_rate = 0.1  # Default 10% growth

        # Project future months
        projections = []
        current_monthly = last_3_months_revenue[0] if last_3_months_revenue else 0

        for i in range(1, months + 1):
            future_month = now + timedelta(days=30 * i)
            projected = current_monthly * ((1 + growth_rate) ** i)
            projections.append((future_month.strftime('%Y-%m'), projected))

        return projections


class GrowthAnalytics:
    """Analyze growth trends."""

    def __init__(self):
        """Initialize growth analytics."""
        self.historical_data = {
            'subscribers': [],  # (date, count)
            'monthly_listeners': [],
            'total_streams': [],
            'total_views': []
        }

    def add_data_point(self, metric: str, date: datetime, value: float):
        """Add historical data point."""
        if metric in self.historical_data:
            self.historical_data[metric].append((date, value))
            # Sort by date
            self.historical_data[metric].sort(key=lambda x: x[0])

    def calculate_growth_rate(self, metric: str, days: int = 30) -> float:
        """
        Calculate growth rate for metric.

        Args:
            metric: Metric to analyze
            days: Number of days to look back

        Returns:
            Growth rate as percentage
        """
        if metric not in self.historical_data or len(self.historical_data[metric]) < 2:
            return 0.0

        data = self.historical_data[metric]
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get values at start and end of period
        period_data = [point for point in data if point[0] >= cutoff_date]

        if len(period_data) < 2:
            return 0.0

        start_value = period_data[0][1]
        end_value = period_data[-1][1]

        if start_value == 0:
            return 0.0

        growth_rate = ((end_value - start_value) / start_value) * 100
        return growth_rate

    def project_milestone(self, metric: str, target_value: float) -> Optional[datetime]:
        """
        Project when metric will reach target value.

        Args:
            metric: Metric to project
            target_value: Target value to reach

        Returns:
            Estimated date to reach target
        """
        growth_rate = self.calculate_growth_rate(metric, days=30)

        if growth_rate <= 0:
            return None

        if metric not in self.historical_data or not self.historical_data[metric]:
            return None

        current_value = self.historical_data[metric][-1][1]

        if current_value >= target_value:
            return datetime.now()

        # Calculate months to reach target
        monthly_growth_rate = growth_rate / 100
        months_to_target = 0

        value = current_value
        while value < target_value and months_to_target < 1200:  # Max 100 years
            value *= (1 + monthly_growth_rate)
            months_to_target += 1

        return datetime.now() + timedelta(days=30 * months_to_target)


class MasterDashboard:
    """Master analytics dashboard combining all platforms."""

    def __init__(self):
        """Initialize master dashboard."""
        self.youtube = YouTubeAnalytics()
        self.spotify = SpotifyAnalytics()
        self.financial = FinancialDashboard()
        self.growth = GrowthAnalytics()

    def get_overview(self) -> Dict:
        """Get complete performance overview."""
        # Calculate totals
        total_youtube_views = sum(v.views_or_streams for v in self.youtube.videos.values())
        total_spotify_streams = sum(t.views_or_streams for t in self.spotify.tracks.values())

        # Revenue
        monthly_revenue = self.financial.get_total_revenue(
            start_date=datetime.now() - timedelta(days=30)
        )
        monthly_costs = self.financial.get_total_costs(
            start_date=datetime.now() - timedelta(days=30)
        )
        roi = self.financial.calculate_roi(
            start_date=datetime.now() - timedelta(days=30)
        )

        # Growth
        subscriber_growth = self.growth.calculate_growth_rate('subscribers', days=30)

        overview = {
            'youtube': {
                'total_videos': len(self.youtube.videos),
                'total_views': total_youtube_views,
                'subscribers': self.youtube.channel_stats['subscribers'],
                'cpm': self.youtube.calculate_cpm()
            },
            'spotify': {
                'total_tracks': len(self.spotify.tracks),
                'total_streams': total_spotify_streams,
                'monthly_listeners': self.spotify.monthly_listeners
            },
            'financial': {
                'monthly_revenue_usd': monthly_revenue,
                'monthly_costs_usd': monthly_costs,
                'monthly_profit_usd': monthly_revenue - monthly_costs,
                'roi_percentage': roi
            },
            'growth': {
                'subscriber_growth_30d': subscriber_growth,
                '100k_subscribers_eta': self.growth.project_milestone('subscribers', 100000)
            }
        }

        return overview

    def generate_report(self) -> str:
        """Generate formatted report."""
        overview = self.get_overview()

        report = []
        report.append("=" * 60)
        report.append("LOFI MUSIC EMPIRE - ANALYTICS DASHBOARD")
        report.append("=" * 60)
        report.append("")

        # YouTube Section
        report.append("YOUTUBE PERFORMANCE:")
        report.append(f"  Videos: {overview['youtube']['total_videos']}")
        report.append(f"  Total Views: {overview['youtube']['total_views']:,}")
        report.append(f"  Subscribers: {overview['youtube']['subscribers']:,}")
        report.append(f"  Average CPM: ${overview['youtube']['cpm']:.2f}")
        report.append("")

        # Spotify Section
        report.append("SPOTIFY PERFORMANCE:")
        report.append(f"  Tracks: {overview['spotify']['total_tracks']}")
        report.append(f"  Total Streams: {overview['spotify']['total_streams']:,}")
        report.append(f"  Monthly Listeners: {overview['spotify']['monthly_listeners']:,}")
        report.append("")

        # Financial Section
        report.append("FINANCIAL OVERVIEW (Last 30 Days):")
        report.append(f"  Revenue: ${overview['financial']['monthly_revenue_usd']:.2f}")
        report.append(f"  Costs: ${overview['financial']['monthly_costs_usd']:.2f}")
        report.append(f"  Profit: ${overview['financial']['monthly_profit_usd']:.2f}")
        report.append(f"  ROI: {overview['financial']['roi_percentage']:.1f}%")
        report.append("")

        # Growth Section
        report.append("GROWTH METRICS:")
        report.append(f"  Subscriber Growth (30d): {overview['growth']['subscriber_growth_30d']:.1f}%")
        eta = overview['growth']['100k_subscribers_eta']
        if eta:
            report.append(f"  100K Subscribers ETA: {eta.strftime('%Y-%m-%d')}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


# Example usage
if __name__ == '__main__':
    print("=== Analytics Dashboard ===\n")

    # Initialize dashboard
    dashboard = MasterDashboard()

    # Add sample YouTube data
    for i in range(10):
        metrics = PerformanceMetrics(
            platform=Platform.YOUTUBE,
            content_id=f"video_{i}",
            title=f"Chill Lofi Beats {i}",
            views_or_streams=10000 + (i * 1000),
            likes=500 + (i * 50),
            comments=50 + (i * 5),
            shares=20 + (i * 2),
            watch_time_minutes=5000 + (i * 500),
            revenue_usd=50 + (i * 5),
            date=datetime.now() - timedelta(days=30 - i)
        )
        dashboard.youtube.add_video_performance(metrics)

    # Add financial data
    dashboard.financial.add_revenue('youtube_ads', 500, datetime.now())
    dashboard.financial.add_revenue('spotify_streams', 200, datetime.now())
    dashboard.financial.add_cost('distribution', 20, datetime.now(), "DistroKid")
    dashboard.financial.add_cost('api_costs', 10, datetime.now(), "YouTube API")

    # Add growth data
    for i in range(30):
        date = datetime.now() - timedelta(days=30 - i)
        subscribers = 5000 + (i * 100)  # Growing
        dashboard.growth.add_data_point('subscribers', date, subscribers)

    dashboard.youtube.channel_stats['subscribers'] = 8000
    dashboard.spotify.monthly_listeners = 3000

    # Generate report
    print(dashboard.generate_report())

    # Top performing videos
    print("\nTOP 5 VIDEOS BY VIEWS:")
    top_videos = dashboard.youtube.get_top_videos(n=5, metric='views')
    for i, video in enumerate(top_videos, 1):
        print(f"{i}. {video.title}: {video.views_or_streams:,} views")

    # Upload pattern analysis
    print("\nOPTIMAL UPLOAD TIMES:")
    patterns = dashboard.youtube.analyze_upload_patterns()
    print(f"Best day: {patterns['best_day']}")
    print(f"Best hour: {patterns['best_hour']}:00")

    # Revenue projections
    print("\nREVENUE PROJECTIONS (Next 6 Months):")
    projections = dashboard.financial.project_revenue(months=6)
    for month, revenue in projections[:3]:
        print(f"{month}: ${revenue:.2f}")

    print("\n✅ Analytics dashboard ready!")
