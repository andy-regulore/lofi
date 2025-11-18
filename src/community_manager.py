"""
Community management automation for audience engagement.

Automates community interaction and growth:
- Comment monitoring and sentiment analysis
- Smart response templates with personalization
- Engagement automation (likes, replies, pins)
- Spam/toxicity detection and filtering
- Community insights and analytics
- Influencer/collaborator discovery
- Fan segmentation (superfans, regulars, new)
- Engagement campaigns
- Community health monitoring

Author: Claude
License: MIT
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from pathlib import Path


class Platform(Enum):
    """Platforms."""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    DISCORD = "discord"


class CommentType(Enum):
    """Comment categories."""
    POSITIVE = "positive"
    QUESTION = "question"
    FEEDBACK = "feedback"
    COLLABORATION = "collaboration"
    SPAM = "spam"
    TOXIC = "toxic"
    NEUTRAL = "neutral"


class UserSegment(Enum):
    """User segments."""
    SUPERFAN = "superfan"          # High engagement, frequent comments
    REGULAR = "regular"             # Regular engagement
    CASUAL = "casual"               # Occasional engagement
    NEW = "new"                     # First-time commenter
    INFLUENCER = "influencer"       # High follower count
    POTENTIAL_COLLAB = "potential_collab"  # Other content creators


@dataclass
class Comment:
    """Comment data."""
    id: str
    platform: Platform
    author: str
    author_id: str
    text: str
    timestamp: datetime
    likes: int
    replies: int
    parent_id: Optional[str] = None
    video_id: Optional[str] = None


@dataclass
class UserProfile:
    """User profile."""
    user_id: str
    username: str
    platform: Platform
    segment: UserSegment
    total_comments: int
    avg_sentiment: float
    first_seen: datetime
    last_seen: datetime
    follower_count: int = 0
    is_verified: bool = False


class SentimentAnalyzer:
    """Analyze comment sentiment."""

    # Keyword lists for simple sentiment analysis
    POSITIVE_KEYWORDS = [
        'love', 'amazing', 'great', 'awesome', 'perfect', 'beautiful',
        'excellent', 'wonderful', 'fantastic', 'best', 'incredible',
        'chill', 'relaxing', 'peaceful', 'vibe', 'mood', '🔥', '❤️', '😍',
        'thank', 'thanks', 'appreciate', 'helpful', 'inspiring'
    ]

    NEGATIVE_KEYWORDS = [
        'hate', 'bad', 'terrible', 'awful', 'worst', 'sucks',
        'boring', 'annoying', 'trash', 'garbage', 'disappointing'
    ]

    QUESTION_KEYWORDS = [
        'how', 'what', 'when', 'where', 'why', 'who', 'which',
        'can you', 'could you', 'would you', '?'
    ]

    SPAM_PATTERNS = [
        r'check out my',
        r'visit my channel',
        r'subscribe to',
        r'click here',
        r'free download',
        r'bit\.ly',
        r'won \$\d+',
        r'congratulations you',
    ]

    @classmethod
    def analyze_sentiment(cls, text: str) -> float:
        """
        Analyze sentiment score.

        Args:
            text: Comment text

        Returns:
            Sentiment score (-1 to 1)
        """
        text_lower = text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for kw in cls.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in cls.NEGATIVE_KEYWORDS if kw in text_lower)

        # Calculate score
        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / total
        return score

    @classmethod
    def classify_comment(cls, text: str) -> CommentType:
        """
        Classify comment type.

        Args:
            text: Comment text

        Returns:
            Comment type
        """
        text_lower = text.lower()

        # Check for spam
        for pattern in cls.SPAM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return CommentType.SPAM

        # Check for toxic content
        if any(word in text_lower for word in ['hate', 'kill', 'die', 'stupid']):
            # Simple toxicity check - in production use Perspective API
            return CommentType.TOXIC

        # Check for questions
        if any(kw in text_lower for kw in cls.QUESTION_KEYWORDS):
            return CommentType.QUESTION

        # Check for collaboration requests
        if any(kw in text_lower for kw in ['collab', 'collaborate', 'work together', 'feature']):
            return CommentType.COLLABORATION

        # Check for feedback
        if any(kw in text_lower for kw in ['suggest', 'idea', 'should', 'could', 'would be better']):
            return CommentType.FEEDBACK

        # Check sentiment
        sentiment = cls.analyze_sentiment(text)
        if sentiment > 0.3:
            return CommentType.POSITIVE
        elif sentiment < -0.3:
            return CommentType.TOXIC

        return CommentType.NEUTRAL


class ResponseTemplates:
    """Smart response templates."""

    TEMPLATES = {
        CommentType.POSITIVE: [
            "Thank you so much! 🙏 Glad you're enjoying the vibes!",
            "Really appreciate the support! ❤️ More coming soon!",
            "Thank you! So happy this resonates with you! 🎵",
            "Thanks for listening! Your support means everything! 🙌",
            "Appreciate you! Hope you have a great study/chill session! ✨",
        ],
        CommentType.QUESTION: [
            "Great question! {answer}",
            "Thanks for asking! {answer}",
            "Good point! {answer}",
        ],
        CommentType.FEEDBACK: [
            "Thanks for the feedback! I'll definitely consider that! 🙏",
            "Great suggestion! Always looking to improve! 💡",
            "Appreciate the input! Will keep that in mind for future tracks! 🎵",
        ],
        CommentType.COLLABORATION: [
            "Thanks for reaching out! Feel free to DM me to discuss! 🤝",
            "Appreciate the interest! Let's connect via email: {email}",
            "Cool! Send me your portfolio/examples and let's chat!",
        ],
        CommentType.NEUTRAL: [
            "Thanks for listening! 🎵",
            "Appreciate you stopping by! 🙏",
            "Hope you enjoy! ✨",
        ],
    }

    PERSONALIZATION_PATTERNS = {
        'username': '{username}',
        'time_of_day': '{time_greeting}',
        'video_title': '{video_title}',
    }

    @classmethod
    def get_response(cls, comment_type: CommentType,
                    personalization: Optional[Dict] = None) -> str:
        """
        Get response template.

        Args:
            comment_type: Comment type
            personalization: Personalization data

        Returns:
            Response text
        """
        import random

        if comment_type not in cls.TEMPLATES:
            return "Thanks for the comment! 🙏"

        # Select random template
        template = random.choice(cls.TEMPLATES[comment_type])

        # Apply personalization
        if personalization:
            for key, value in personalization.items():
                template = template.replace(f'{{{key}}}', str(value))

        return template

    @classmethod
    def personalize_response(cls, template: str, comment: Comment,
                           user_profile: Optional[UserProfile] = None) -> str:
        """
        Personalize response template.

        Args:
            template: Template text
            comment: Comment data
            user_profile: User profile

        Returns:
            Personalized response
        """
        # Add username if superfan
        if user_profile and user_profile.segment == UserSegment.SUPERFAN:
            if '{username}' not in template:
                template = f"Hey {comment.author}! " + template

        # Add time-appropriate greeting
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 18:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        template = template.replace('{time_greeting}', time_greeting)

        return template


class UserSegmenter:
    """Segment users by behavior."""

    def __init__(self):
        """Initialize user segmenter."""
        self.user_profiles: Dict[str, UserProfile] = {}

    def update_user_profile(self, comment: Comment):
        """
        Update user profile based on comment.

        Args:
            comment: Comment data
        """
        user_id = comment.author_id

        if user_id not in self.user_profiles:
            # Create new profile
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                username=comment.author,
                platform=comment.platform,
                segment=UserSegment.NEW,
                total_comments=1,
                avg_sentiment=SentimentAnalyzer.analyze_sentiment(comment.text),
                first_seen=comment.timestamp,
                last_seen=comment.timestamp
            )
        else:
            # Update existing profile
            profile = self.user_profiles[user_id]
            profile.total_comments += 1
            profile.last_seen = comment.timestamp

            # Update average sentiment
            new_sentiment = SentimentAnalyzer.analyze_sentiment(comment.text)
            profile.avg_sentiment = (profile.avg_sentiment * (profile.total_comments - 1) + new_sentiment) / profile.total_comments

            # Re-segment
            profile.segment = self._determine_segment(profile)

    def _determine_segment(self, profile: UserProfile) -> UserSegment:
        """
        Determine user segment.

        Args:
            profile: User profile

        Returns:
            User segment
        """
        # Check if verified/influencer
        if profile.is_verified or profile.follower_count > 10000:
            return UserSegment.INFLUENCER

        # Check engagement frequency
        days_active = (profile.last_seen - profile.first_seen).days + 1
        comments_per_day = profile.total_comments / days_active

        if profile.total_comments >= 10 and comments_per_day > 0.5:
            return UserSegment.SUPERFAN
        elif profile.total_comments >= 3:
            return UserSegment.REGULAR
        elif profile.total_comments > 1:
            return UserSegment.CASUAL
        else:
            return UserSegment.NEW

    def get_superfans(self, limit: int = 10) -> List[UserProfile]:
        """
        Get top superfans.

        Args:
            limit: Number of superfans to return

        Returns:
            List of superfan profiles
        """
        superfans = [p for p in self.user_profiles.values() if p.segment == UserSegment.SUPERFAN]
        superfans.sort(key=lambda x: x.total_comments, reverse=True)
        return superfans[:limit]

    def get_potential_collabs(self) -> List[UserProfile]:
        """
        Get potential collaboration candidates.

        Returns:
            List of potential collaborators
        """
        return [p for p in self.user_profiles.values() if p.segment == UserSegment.INFLUENCER or p.follower_count > 5000]


class EngagementBot:
    """Automated engagement actions."""

    def __init__(self, dry_run: bool = True):
        """
        Initialize engagement bot.

        Args:
            dry_run: If True, only log actions without executing
        """
        self.dry_run = dry_run
        self.response_templates = ResponseTemplates()

    def should_respond(self, comment: Comment, comment_type: CommentType) -> bool:
        """
        Determine if should respond to comment.

        Args:
            comment: Comment data
            comment_type: Comment type

        Returns:
            True if should respond
        """
        # Always respond to questions and collaboration requests
        if comment_type in [CommentType.QUESTION, CommentType.COLLABORATION]:
            return True

        # Never respond to spam or toxic
        if comment_type in [CommentType.SPAM, CommentType.TOXIC]:
            return False

        # Respond to positive comments with some probability
        if comment_type == CommentType.POSITIVE:
            # Respond to 30% of positive comments
            import random
            return random.random() < 0.3

        # Respond to feedback
        if comment_type == CommentType.FEEDBACK:
            return True

        return False

    def generate_response(self, comment: Comment, comment_type: CommentType,
                         user_profile: Optional[UserProfile] = None) -> str:
        """
        Generate response for comment.

        Args:
            comment: Comment data
            comment_type: Comment type
            user_profile: User profile

        Returns:
            Response text
        """
        # Get base template
        response = self.response_templates.get_response(comment_type)

        # Personalize
        response = self.response_templates.personalize_response(
            response, comment, user_profile
        )

        return response

    def post_response(self, comment: Comment, response_text: str):
        """
        Post response to comment.

        Args:
            comment: Original comment
            response_text: Response text
        """
        if self.dry_run:
            print(f"[DRY RUN] Would reply to '{comment.text[:50]}...'")
            print(f"          Response: '{response_text}'")
        else:
            # In production: use platform API to post reply
            # youtube.comments().insert(...)
            # instagram.media().comments().create(...)
            pass

    def like_comment(self, comment: Comment):
        """
        Like a comment.

        Args:
            comment: Comment to like
        """
        if self.dry_run:
            print(f"[DRY RUN] Would like comment by {comment.author}")
        else:
            # In production: use platform API to like
            pass

    def pin_comment(self, comment: Comment):
        """
        Pin a comment.

        Args:
            comment: Comment to pin
        """
        if self.dry_run:
            print(f"[DRY RUN] Would pin comment by {comment.author}")
        else:
            # In production: use platform API to pin
            pass


class CommunityAnalytics:
    """Community insights and analytics."""

    def __init__(self):
        """Initialize community analytics."""
        self.comments: List[Comment] = []

    def add_comment(self, comment: Comment):
        """Add comment to analytics."""
        self.comments.append(comment)

    def get_engagement_rate(self, video_views: int) -> float:
        """
        Calculate engagement rate.

        Args:
            video_views: Total video views

        Returns:
            Engagement rate
        """
        if video_views == 0:
            return 0.0

        total_comments = len(self.comments)
        return (total_comments / video_views) * 100

    def get_sentiment_distribution(self) -> Dict[str, float]:
        """
        Get sentiment distribution.

        Returns:
            Dict with sentiment percentages
        """
        if not self.comments:
            return {'positive': 0, 'neutral': 0, 'negative': 0}

        sentiments = [SentimentAnalyzer.analyze_sentiment(c.text) for c in self.comments]

        positive = sum(1 for s in sentiments if s > 0.3)
        negative = sum(1 for s in sentiments if s < -0.3)
        neutral = len(sentiments) - positive - negative

        total = len(sentiments)

        return {
            'positive': (positive / total) * 100,
            'neutral': (neutral / total) * 100,
            'negative': (negative / total) * 100
        }

    def get_comment_type_distribution(self) -> Dict[CommentType, int]:
        """
        Get comment type distribution.

        Returns:
            Dict mapping comment type to count
        """
        distribution = {}

        for comment in self.comments:
            comment_type = SentimentAnalyzer.classify_comment(comment.text)
            distribution[comment_type] = distribution.get(comment_type, 0) + 1

        return distribution

    def get_peak_activity_times(self) -> Dict[int, int]:
        """
        Get peak comment activity times.

        Returns:
            Dict mapping hour to comment count
        """
        hour_counts = {hour: 0 for hour in range(24)}

        for comment in self.comments:
            hour = comment.timestamp.hour
            hour_counts[hour] += 1

        return hour_counts

    def get_top_commenters(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get top commenters by volume.

        Args:
            limit: Number of top commenters

        Returns:
            List of (username, comment_count) tuples
        """
        commenter_counts = {}

        for comment in self.comments:
            commenter_counts[comment.author] = commenter_counts.get(comment.author, 0) + 1

        # Sort and return top N
        sorted_commenters = sorted(commenter_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_commenters[:limit]


class CommunityManager:
    """Main community management system."""

    def __init__(self, dry_run: bool = True):
        """
        Initialize community manager.

        Args:
            dry_run: If True, simulate actions without executing
        """
        self.dry_run = dry_run
        self.segmenter = UserSegmenter()
        self.bot = EngagementBot(dry_run=dry_run)
        self.analytics = CommunityAnalytics()
        self.processed_comments = set()

    def process_comment(self, comment: Comment):
        """
        Process new comment.

        Args:
            comment: Comment data
        """
        # Skip if already processed
        if comment.id in self.processed_comments:
            return

        # Add to analytics
        self.analytics.add_comment(comment)

        # Update user profile
        self.segmenter.update_user_profile(comment)
        user_profile = self.segmenter.user_profiles.get(comment.author_id)

        # Classify comment
        comment_type = SentimentAnalyzer.classify_comment(comment.text)

        print(f"\n[{comment.platform.value}] {comment.author}: {comment.text[:60]}...")
        print(f"  Type: {comment_type.value}")
        print(f"  Segment: {user_profile.segment.value if user_profile else 'unknown'}")

        # Handle spam/toxic
        if comment_type == CommentType.SPAM:
            print("  Action: Mark as spam")
            # In production: report/delete
            self.processed_comments.add(comment.id)
            return

        if comment_type == CommentType.TOXIC:
            print("  Action: Hide/review")
            # In production: hide comment for review
            self.processed_comments.add(comment.id)
            return

        # Like comment
        if comment_type in [CommentType.POSITIVE, CommentType.FEEDBACK, CommentType.QUESTION]:
            self.bot.like_comment(comment)

        # Pin superfan comments
        if user_profile and user_profile.segment == UserSegment.SUPERFAN:
            if comment_type == CommentType.POSITIVE:
                # Maybe pin (don't pin every superfan comment)
                import random
                if random.random() < 0.1:
                    self.bot.pin_comment(comment)

        # Generate and post response
        if self.bot.should_respond(comment, comment_type):
            response = self.bot.generate_response(comment, comment_type, user_profile)
            self.bot.post_response(comment, response)

        self.processed_comments.add(comment.id)

    def get_community_insights(self) -> Dict:
        """
        Get comprehensive community insights.

        Returns:
            Insights dict
        """
        insights = {
            'total_comments': len(self.analytics.comments),
            'sentiment_distribution': self.analytics.get_sentiment_distribution(),
            'comment_types': self.analytics.get_comment_type_distribution(),
            'peak_hours': self.analytics.get_peak_activity_times(),
            'top_commenters': self.analytics.get_top_commenters(10),
            'superfans': len(self.segmenter.get_superfans()),
            'potential_collabs': len(self.segmenter.get_potential_collabs()),
        }

        return insights


# Example usage
if __name__ == '__main__':
    print("=== Community Management System ===\n")

    # Initialize manager
    manager = CommunityManager(dry_run=True)

    # Simulate comments
    sample_comments = [
        Comment("1", Platform.YOUTUBE, "StudyBuddy", "user1",
                "This is perfect for studying! Love the vibe 🎵",
                datetime.now(), 15, 0),
        Comment("2", Platform.YOUTUBE, "MusicFan", "user2",
                "How do you make these beats? Any tips?",
                datetime.now(), 5, 2),
        Comment("3", Platform.YOUTUBE, "ChillSeeker", "user3",
                "So relaxing ❤️ been listening for hours",
                datetime.now(), 8, 0),
        Comment("4", Platform.YOUTUBE, "ProducerJoe", "user4",
                "Hey, I'm a producer too. Would love to collab!",
                datetime.now(), 3, 0),
        Comment("5", Platform.YOUTUBE, "SpamBot", "user5",
                "Check out my channel for free beats! Click here: bit.ly/spam",
                datetime.now(), 0, 0),
    ]

    print("1. Processing Comments:")
    for comment in sample_comments:
        manager.process_comment(comment)

    print("\n" + "="*60)
    print("\n2. Community Insights:")
    insights = manager.get_community_insights()
    print(f"Total comments: {insights['total_comments']}")
    print(f"\nSentiment distribution:")
    for sentiment, pct in insights['sentiment_distribution'].items():
        print(f"  {sentiment}: {pct:.1f}%")

    print(f"\nComment types:")
    for comment_type, count in insights['comment_types'].items():
        print(f"  {comment_type.value}: {count}")

    print(f"\nCommunity health:")
    print(f"  Superfans: {insights['superfans']}")
    print(f"  Potential collabs: {insights['potential_collabs']}")

    print("\n✅ Community management system ready!")
    print("   Automated responses configured")
    print("   Spam detection active")
    print("   Community insights available")
