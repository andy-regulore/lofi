"""
Social Media Automation for Multi-Platform Music Promotion

Automates posting to:
- Instagram (posts, stories, reels)
- TikTok (short-form videos)
- Twitter (tweets with previews)
- Reddit (community engagement)

Author: Claude
License: MIT
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import time


class SocialMediaManager:
    """Unified social media management across platforms."""

    def __init__(self, config: dict):
        """
        Initialize social media manager.

        Args:
            config: Configuration dict with API keys
        """
        self.config = config
        self.instagram = InstagramBot(config.get('instagram', {}))
        self.tiktok = TikTokBot(config.get('tiktok', {}))
        self.twitter = TwitterBot(config.get('twitter', {}))
        self.reddit = RedditBot(config.get('reddit', {}))

    def promote_new_track(
        self,
        track_info: Dict,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Promote new track across all social platforms.

        Args:
            track_info: Track information (title, audio_path, video_path, etc.)
            platforms: List of platforms or None for all

        Returns:
            Results dict per platform
        """
        if platforms is None:
            platforms = ['instagram', 'tiktok', 'twitter', 'reddit']

        results = {}

        print(f"📱 Promoting: {track_info.get('title')}")

        # Instagram
        if 'instagram' in platforms:
            results['instagram'] = self.instagram.post_track(track_info)

        # TikTok
        if 'tiktok' in platforms:
            results['tiktok'] = self.tiktok.post_track(track_info)

        # Twitter
        if 'twitter' in platforms:
            results['twitter'] = self.twitter.post_track(track_info)

        # Reddit
        if 'reddit' in platforms:
            results['reddit'] = self.reddit.post_track(track_info)

        return results

    def schedule_posts(
        self,
        track_list: List[Dict],
        posts_per_day: int = 3,
        days: int = 7
    ) -> List[Dict]:
        """
        Create scheduled posting calendar.

        Args:
            track_list: List of tracks to promote
            posts_per_day: Posts per day
            days: Number of days to schedule

        Returns:
            List of scheduled post dicts
        """
        schedule = []
        post_times = self._generate_optimal_times(posts_per_day)

        track_index = 0

        for day in range(days):
            for post_time in post_times:
                if track_index >= len(track_list):
                    break

                post_datetime = datetime.now() + timedelta(
                    days=day,
                    hours=post_time['hour'],
                    minutes=post_time['minute']
                )

                schedule.append({
                    'track': track_list[track_index],
                    'scheduled_time': post_datetime.isoformat(),
                    'platforms': ['instagram', 'twitter'],  # Customizable
                    'status': 'pending'
                })

                track_index += 1

        return schedule

    def _generate_optimal_times(self, posts_per_day: int) -> List[Dict]:
        """Generate optimal posting times."""
        # Best times for music content (based on engagement data)
        optimal_hours = [9, 14, 20]  # 9 AM, 2 PM, 8 PM

        times = []
        for i in range(posts_per_day):
            hour = optimal_hours[i % len(optimal_hours)]
            times.append({'hour': hour, 'minute': 0})

        return times


class InstagramBot:
    """Instagram automation."""

    def __init__(self, config: dict):
        """Initialize Instagram bot."""
        self.access_token = config.get('access_token')
        self.user_id = config.get('user_id')
        self.api_url = 'https://graph.instagram.com'

    def post_track(self, track_info: Dict) -> Dict:
        """
        Post track to Instagram.

        Args:
            track_info: Track information

        Returns:
            Post result
        """
        if not self.access_token:
            return {
                'status': 'error',
                'message': 'Instagram access token not configured',
                'setup_instructions': [
                    '1. Create Facebook App',
                    '2. Add Instagram Graph API',
                    '3. Get user access token',
                    '4. Add to config.json'
                ]
            }

        # Post to feed
        feed_result = self._post_to_feed(track_info)

        # Post to story
        story_result = self._post_to_story(track_info)

        # Post reel (if video available)
        reel_result = None
        if track_info.get('video_path'):
            reel_result = self._post_reel(track_info)

        return {
            'platform': 'instagram',
            'feed_post': feed_result,
            'story': story_result,
            'reel': reel_result
        }

    def _post_to_feed(self, track_info: Dict) -> Dict:
        """Post to Instagram feed."""
        # Create media container
        caption = self._generate_caption(track_info)

        # Upload image/video
        # This requires Instagram Graph API and proper authentication

        return {
            'status': 'manual_required',
            'message': 'Use Instagram Graph API or post manually',
            'caption': caption,
            'hashtags': self._generate_hashtags(track_info)
        }

    def _post_to_story(self, track_info: Dict) -> Dict:
        """Post to Instagram story."""
        return {
            'status': 'manual_required',
            'message': 'Stories require manual posting or third-party service'
        }

    def _post_reel(self, track_info: Dict) -> Dict:
        """Post reel."""
        return {
            'status': 'manual_required',
            'message': 'Reels require manual posting',
            'video_path': track_info.get('video_path')
        }

    def _generate_caption(self, track_info: Dict) -> str:
        """Generate Instagram caption."""
        title = track_info.get('title', 'New LoFi Track')
        mood = track_info.get('mood', 'chill')

        captions = [
            f"🎵 {title}\n\n{mood.capitalize()} vibes for your day ✨\n\nLink in bio for full track!",
            f"New {mood} beats just dropped 🔥\n\n{title}\n\nPerfect for studying, working, or just vibing 🎧",
            f"✨ {title} ✨\n\nYour new {mood} soundtrack is here\n\nStream now - link in bio 🎶"
        ]

        import random
        return random.choice(captions)

    def _generate_hashtags(self, track_info: Dict) -> List[str]:
        """Generate relevant hashtags."""
        mood = track_info.get('mood', 'chill')

        base_hashtags = [
            '#lofi', '#lofibeats', '#lofihiphop', '#chillbeats',
            '#studymusic', '#relaxingmusic', '#instrumental',
            '#beats', '#producer', '#musicproducer'
        ]

        mood_hashtags = {
            'chill': ['#chillvibes', '#chillmusic', '#chillhop'],
            'focus': ['#focusmusic', '#studybeats', '#concentration'],
            'happy': ['#happybeats', '#goodvibes', '#positivevibes'],
            'peaceful': ['#peaceful', '#calm', '#meditation']
        }

        hashtags = base_hashtags + mood_hashtags.get(mood, [])
        return hashtags[:30]  # Instagram limit


class TikTokBot:
    """TikTok automation."""

    def __init__(self, config: dict):
        """Initialize TikTok bot."""
        self.access_token = config.get('access_token')

    def post_track(self, track_info: Dict) -> Dict:
        """
        Post track to TikTok.

        Args:
            track_info: Track information

        Returns:
            Post result
        """
        if not track_info.get('video_path'):
            return {
                'status': 'error',
                'message': 'Video required for TikTok'
            }

        # TikTok requires short-form video (15-60s)
        caption = self._generate_caption(track_info)

        return {
            'status': 'manual_required',
            'platform': 'tiktok',
            'message': 'TikTok API requires approval - post manually or use scheduling service',
            'video_path': track_info['video_path'],
            'caption': caption,
            'hashtags': self._generate_hashtags(track_info),
            'instructions': [
                '1. Open TikTok app',
                '2. Upload video',
                '3. Add caption and hashtags',
                '4. Post'
            ]
        }

    def _generate_caption(self, track_info: Dict) -> str:
        """Generate TikTok caption."""
        title = track_info.get('title', 'New LoFi Track')
        mood = track_info.get('mood', 'chill')

        captions = [
            f"{mood} beats to vibe to 🎧 #{mood}lofi #beats",
            f"New {title} 🔥 Perfect for studying! #lofi #studybeats",
            f"POV: You're studying at 2 AM 📚✨ #lofi #latenight"
        ]

        import random
        return random.choice(captions)

    def _generate_hashtags(self, track_info: Dict) -> List[str]:
        """Generate TikTok hashtags."""
        return [
            '#lofi', '#lofibeats', '#chillbeats', '#studymusic',
            '#fyp', '#foryoupage', '#beats', '#music',
            '#studying', '#chill', '#vibe'
        ]


class TwitterBot:
    """Twitter automation."""

    def __init__(self, config: dict):
        """Initialize Twitter bot."""
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.access_token = config.get('access_token')
        self.access_secret = config.get('access_secret')

    def post_track(self, track_info: Dict) -> Dict:
        """
        Post track to Twitter.

        Args:
            track_info: Track information

        Returns:
            Post result
        """
        if not self.api_key:
            return {
                'status': 'error',
                'message': 'Twitter API keys not configured',
                'setup_instructions': [
                    '1. Apply for Twitter Developer account',
                    '2. Create app',
                    '3. Get API keys',
                    '4. Add to config.json'
                ]
            }

        tweet_text = self._generate_tweet(track_info)

        try:
            # Using tweepy library (if installed)
            import tweepy

            auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
            auth.set_access_token(self.access_token, self.access_secret)
            api = tweepy.API(auth)

            # Post tweet
            tweet = api.update_status(tweet_text)

            return {
                'status': 'success',
                'platform': 'twitter',
                'tweet_id': tweet.id_str,
                'url': f"https://twitter.com/user/status/{tweet.id_str}"
            }

        except ImportError:
            return {
                'status': 'error',
                'message': 'Tweepy not installed. Run: pip install tweepy',
                'tweet_text': tweet_text
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'tweet_text': tweet_text
            }

    def _generate_tweet(self, track_info: Dict) -> str:
        """Generate tweet text."""
        title = track_info.get('title', 'New Track')
        mood = track_info.get('mood', 'chill')
        youtube_url = track_info.get('youtube_url', '')

        tweets = [
            f"🎵 New {mood} beats: {title}\n\nPerfect for studying, working, or just vibing ✨\n\n{youtube_url}\n\n#lofi #beats",
            f"Just dropped: {title} 🔥\n\nYour new {mood} soundtrack is live\n\n{youtube_url}\n\n#lofibeats #chillhop",
            f"New track alert! 🎧\n\n{title}\n\n{mood.capitalize()} vibes for your day\n\n{youtube_url}"
        ]

        import random
        return random.choice(tweets)


class RedditBot:
    """Reddit automation for community engagement."""

    def __init__(self, config: dict):
        """Initialize Reddit bot."""
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.username = config.get('username')
        self.password = config.get('password')

    def post_track(self, track_info: Dict) -> Dict:
        """
        Post track to relevant subreddits.

        Args:
            track_info: Track information

        Returns:
            Post result
        """
        if not self.client_id:
            return {
                'status': 'error',
                'message': 'Reddit API credentials not configured'
            }

        # Relevant subreddits for LoFi
        subreddits = ['LofiHipHop', 'chillmusic', 'StudyMusic', 'LofiGirl']

        # Note: Reddit has strict self-promotion rules
        # Only post if you're an active community member

        return {
            'status': 'manual_required',
            'platform': 'reddit',
            'message': 'Reddit requires manual posting (anti-spam rules)',
            'recommended_subreddits': subreddits,
            'title': track_info.get('title'),
            'url': track_info.get('youtube_url', ''),
            'guidelines': [
                '1. Be an active community member first',
                '2. Follow 10:1 rule (10 comments per 1 promotion)',
                '3. Read subreddit rules before posting',
                '4. Engage with comments on your post'
            ]
        }


def quick_promote(
    track_info: Dict,
    config_path: str = 'config.json'
) -> Dict:
    """
    Quick function to promote track across all platforms.

    Args:
        track_info: Track information
        config_path: Path to config file

    Returns:
        Promotion results
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Create manager
    manager = SocialMediaManager(config)

    # Promote
    results = manager.promote_new_track(track_info)

    return results


if __name__ == '__main__':
    print("📱 Social Media Automation")
    print("=" * 60)

    demo_track = {
        'title': 'Midnight Study Session',
        'mood': 'chill',
        'audio_path': 'output/audio/track.wav',
        'video_path': 'output/videos/track.mp4',
        'thumbnail_path': 'output/thumbnails/track.png',
        'youtube_url': 'https://youtube.com/watch?v=...'
    }

    print("\nDemo track:", demo_track['title'])
    print("\nTo enable automation:")
    print("1. Add API keys to config.json for each platform")
    print("2. Run: quick_promote(track_info)")
    print("\nFor now, we'll generate content for manual posting...")

    # Show what would be generated
    manager = SocialMediaManager({})

    print("\n📸 Instagram Caption:")
    print(manager.instagram._generate_caption(demo_track))

    print("\n🎵 TikTok Caption:")
    print(manager.tiktok._generate_caption(demo_track))

    print("\n🐦 Twitter Tweet:")
    print(manager.twitter._generate_tweet(demo_track))
