"""
AI-powered metadata generator for music tracks.

Generates SEO-optimized titles, descriptions, and tags for:
- YouTube uploads
- Spotify/Apple Music releases
- Social media posts
- Playlist organization

Features:
- Title generation with mood/style keywords
- SEO-optimized descriptions with timestamps
- Trending tag research and optimization
- Platform-specific formatting
- A/B testing variations

Author: Claude
License: MIT
"""

import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackMetadata:
    """Complete metadata for a track."""

    title: str
    description: str
    tags: List[str]
    category: str
    mood: str
    style: str
    bpm: int
    key: str
    duration_seconds: int


class MetadataGenerator:
    """Generate SEO-optimized metadata for music tracks."""

    # Title templates for different moods
    TITLE_TEMPLATES = {
        "chill": [
            "Chill {style} Beats to {activity} [{mood}]",
            "{mood} {style} Mix - {time_of_day} Vibes",
            "Relaxing {style} | {mood} Study Session",
            "{style} Beats for {activity} | {mood} Atmosphere",
            "{time_of_day} {style} - {mood} Instrumental Mix",
        ],
        "study": [
            "{style} Study Beats - {mood} Focus Music",
            "Study With Me | {style} {mood} Beats",
            "Focus Music - {style} for {activity} [{mood}]",
            "{mood} Study Session | {style} Concentration",
            "Productive {style} Beats to {activity} [{mood}]",
        ],
        "sleep": [
            "{style} for Sleep | {mood} Night Sounds",
            "Sleep Music - {mood} {style} Ambience",
            "{mood} {style} for Deep Sleep and Relaxation",
            "Peaceful {style} | {mood} Sleeping Sounds",
            "Night {style} - {mood} Dream Music",
        ],
        "relax": [
            "Relaxing {style} Music | {mood} Chill Beats",
            "{mood} {style} for Relaxation and Peace",
            "Chill {style} Mix - {mood} Relaxing Vibes",
            "Peaceful {style} | {mood} Stress Relief",
            "{mood} Relaxation Music | {style} Ambience",
        ],
        "work": [
            "{style} for Working | {mood} Productivity Beats",
            "Work Music - {mood} {style} for Focus",
            "Productive {style} | {mood} Background Music",
            "{mood} {style} to Boost Productivity",
            "Focus Flow - {style} Work Music [{mood}]",
        ],
    }

    # Activity keywords
    ACTIVITIES = [
        "Study",
        "Work",
        "Read",
        "Focus",
        "Relax",
        "Sleep",
        "Chill",
        "Code",
        "Write",
        "Draw",
        "Create",
        "Think",
        "Meditate",
    ]

    # Time of day
    TIME_OF_DAY = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Late Night",
        "Midnight",
        "Dawn",
        "Sunset",
        "Sunrise",
    ]

    # Mood descriptors
    MOODS = {
        "peaceful": ["Peaceful", "Serene", "Calm", "Tranquil", "Quiet"],
        "melancholic": ["Melancholic", "Nostalgic", "Reflective", "Wistful", "Pensive"],
        "uplifting": ["Uplifting", "Inspiring", "Hopeful", "Bright", "Positive"],
        "dreamy": ["Dreamy", "Ethereal", "Ambient", "Floating", "Hazy"],
        "cozy": ["Cozy", "Warm", "Comfortable", "Intimate", "Soft"],
        "energetic": ["Energetic", "Upbeat", "Lively", "Dynamic", "Vibrant"],
    }

    # Style descriptors
    STYLES = {
        "lofi": ["Lo-Fi Hip Hop", "Chill Hop", "Jazz Hop", "Lo-Fi Beats", "Chillhop"],
        "jazz": ["Jazz", "Jazzy", "Jazz-influenced", "Smooth Jazz", "Neo-Jazz"],
        "ambient": ["Ambient", "Atmospheric", "Soundscape", "Drone", "Minimalist"],
        "electronic": ["Electronic", "Downtempo", "Chillwave", "Synth", "Future Beats"],
        "classical": ["Classical", "Neoclassical", "Piano", "Orchestral", "Chamber"],
    }

    # SEO tags for LoFi
    BASE_TAGS = [
        "lofi",
        "lofi hip hop",
        "chill beats",
        "study music",
        "focus music",
        "relaxing music",
        "chill music",
        "background music",
        "instrumental",
        "beats to study to",
        "beats to relax to",
        "calm music",
        "peaceful music",
    ]

    # Trending tags (updated periodically)
    TRENDING_TAGS = [
        "study with me",
        "work from home",
        "productivity music",
        "deep focus",
        "concentration music",
        "reading music",
        "coding music",
        "writing music",
        "stress relief",
        "anxiety relief",
        "sleep music",
        "meditation music",
    ]

    # Seasonal tags
    SEASONAL_TAGS = {
        "winter": ["winter vibes", "cozy winter", "winter study", "snow ambience"],
        "spring": ["spring vibes", "spring awakening", "fresh sounds", "renewal"],
        "summer": ["summer vibes", "sunny study", "summer chill", "beach sounds"],
        "fall": ["autumn vibes", "fall aesthetic", "cozy autumn", "rainy days"],
    }

    def __init__(self):
        """Initialize metadata generator."""
        self.generated_titles = set()  # Track to avoid duplicates

    def generate_title(
        self, mood: str = "chill", style: str = "lofi", use_case: str = "study", variation: int = 0
    ) -> str:
        """
        Generate SEO-optimized title.

        Args:
            mood: Track mood
            style: Music style
            use_case: Primary use case (study, sleep, work, etc.)
            variation: Template variation number

        Returns:
            Generated title
        """
        # Get template
        templates = self.TITLE_TEMPLATES.get(use_case, self.TITLE_TEMPLATES["chill"])
        template = templates[variation % len(templates)]

        # Select descriptors
        mood_word = random.choice(self.MOODS.get(mood, [mood.title()]))
        style_word = random.choice(self.STYLES.get(style, [style.title()]))
        activity = random.choice(self.ACTIVITIES)
        time = random.choice(self.TIME_OF_DAY)

        # Format title
        title = template.format(
            mood=mood_word, style=style_word, activity=activity, time_of_day=time
        )

        # Ensure uniqueness
        if title in self.generated_titles:
            # Add date or variant
            title = f"{title} ({datetime.now().strftime('%b %Y')})"

        self.generated_titles.add(title)
        return title

    def generate_description(
        self,
        title: str,
        duration: int,
        mood: str,
        style: str,
        bpm: int,
        key: str,
        include_timestamps: bool = True,
        include_cta: bool = True,
    ) -> str:
        """
        Generate SEO-optimized description.

        Args:
            title: Track title
            duration: Duration in seconds
            mood: Track mood
            style: Music style
            bpm: Tempo in BPM
            key: Musical key
            include_timestamps: Include timestamp sections
            include_cta: Include call-to-action

        Returns:
            Generated description
        """
        description_parts = []

        # Opening hook
        hooks = [
            f"Welcome to your ultimate {mood} {style} experience!",
            f"Dive into this {mood} {style} mix perfect for focus and relaxation.",
            f"Enjoy this carefully curated {style} session designed for {mood} vibes.",
            f"Immerse yourself in {duration//60} minutes of pure {mood} {style}.",
        ]
        description_parts.append(random.choice(hooks))
        description_parts.append("")

        # Track details
        description_parts.append("🎵 TRACK INFO:")
        description_parts.append(f"• Style: {style.title()}")
        description_parts.append(f"• Mood: {mood.title()}")
        description_parts.append(f"• BPM: {bpm}")
        description_parts.append(f"• Key: {key}")
        description_parts.append(f"• Duration: {duration//60}:{duration%60:02d}")
        description_parts.append("")

        # Perfect for section
        use_cases = [
            "📚 Studying and homework",
            "💼 Working and productivity",
            "📖 Reading and writing",
            "🧘 Meditation and relaxation",
            "😴 Sleep and rest",
            "🎨 Creative work and art",
            "💻 Coding and programming",
            "☕ Coffee shop ambience",
        ]
        description_parts.append("✨ PERFECT FOR:")
        for use_case in random.sample(use_cases, 4):
            description_parts.append(use_case)
        description_parts.append("")

        # Timestamps (if requested)
        if include_timestamps and duration > 300:
            description_parts.append("⏱️ TIMESTAMPS:")
            num_segments = min(duration // 300, 10)  # Max 10 segments
            for i in range(num_segments):
                timestamp = i * (duration // num_segments)
                minutes = timestamp // 60
                seconds = timestamp % 60
                segment_title = self._generate_segment_title(mood, style, i)
                description_parts.append(f"{minutes}:{seconds:02d} - {segment_title}")
            description_parts.append("")

        # Call to action (if requested)
        if include_cta:
            ctas = [
                "🔔 Subscribe for daily chill beats!",
                "👍 Like if this helps you focus!",
                "💬 Comment your study goals below!",
                "🔔 Turn on notifications for new uploads!",
                "📌 Save to your study playlist!",
            ]
            description_parts.append(random.choice(ctas))
            description_parts.append("")

        # Tags
        description_parts.append("🏷️ TAGS:")
        tags = self.generate_tags(mood, style, use_case="study")
        description_parts.append(", ".join(tags[:10]))
        description_parts.append("")

        # Footer
        description_parts.append("---")
        description_parts.append("All music produced with ❤️ for your focus and relaxation.")
        description_parts.append(f"© {datetime.now().year} | All Rights Reserved")

        return "\n".join(description_parts)

    def _generate_segment_title(self, mood: str, style: str, index: int) -> str:
        """Generate title for timestamp segment."""
        segment_themes = [
            "Opening Vibes",
            "Flow State",
            "Deep Focus",
            "Gentle Groove",
            "Peaceful Moment",
            "Contemplation",
            "Smooth Transition",
            "Energy Shift",
            "Reflection",
            "Building Momentum",
            "Calm Waters",
            "Final Thoughts",
        ]
        return segment_themes[index % len(segment_themes)]

    def generate_tags(
        self,
        mood: str,
        style: str,
        use_case: str = "study",
        seasonal: Optional[str] = None,
        max_tags: int = 30,
    ) -> List[str]:
        """
        Generate SEO-optimized tags.

        Args:
            mood: Track mood
            style: Music style
            use_case: Primary use case
            seasonal: Season (winter, spring, summer, fall)
            max_tags: Maximum number of tags

        Returns:
            List of tags
        """
        tags = []

        # Base tags
        tags.extend(self.BASE_TAGS)

        # Style-specific tags
        if style.lower() in self.STYLES:
            tags.extend([s.lower() for s in self.STYLES[style.lower()]])

        # Mood-specific tags
        if mood.lower() in self.MOODS:
            tags.extend([m.lower() for m in self.MOODS[mood.lower()]])

        # Use case tags
        use_case_tags = {
            "study": ["study music", "focus music", "concentration", "homework music"],
            "sleep": ["sleep music", "sleeping", "bedtime music", "night sounds"],
            "work": ["work music", "productivity", "office music", "background work"],
            "relax": ["relaxation", "chill out", "stress relief", "calm down"],
        }
        tags.extend(use_case_tags.get(use_case, []))

        # Trending tags
        tags.extend(random.sample(self.TRENDING_TAGS, min(5, len(self.TRENDING_TAGS))))

        # Seasonal tags
        if seasonal and seasonal in self.SEASONAL_TAGS:
            tags.extend(self.SEASONAL_TAGS[seasonal])

        # Remove duplicates and limit
        unique_tags = []
        seen = set()
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                unique_tags.append(tag)
                seen.add(tag_lower)

        return unique_tags[:max_tags]

    def generate_complete_metadata(
        self,
        mood: str = "chill",
        style: str = "lofi",
        use_case: str = "study",
        bpm: int = 75,
        key: str = "Am",
        duration: int = 3600,
        seasonal: Optional[str] = None,
    ) -> TrackMetadata:
        """
        Generate complete metadata for a track.

        Args:
            mood: Track mood
            style: Music style
            use_case: Primary use case
            bpm: Tempo
            key: Musical key
            duration: Duration in seconds
            seasonal: Season

        Returns:
            Complete TrackMetadata object
        """
        title = self.generate_title(mood, style, use_case)
        description = self.generate_description(title, duration, mood, style, bpm, key)
        tags = self.generate_tags(mood, style, use_case, seasonal)

        return TrackMetadata(
            title=title,
            description=description,
            tags=tags,
            category="Music",
            mood=mood,
            style=style,
            bpm=bpm,
            key=key,
            duration_seconds=duration,
        )

    def generate_ab_test_variations(
        self, mood: str, style: str, use_case: str, num_variations: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple title/thumbnail variations for A/B testing.

        Args:
            mood: Track mood
            style: Music style
            use_case: Primary use case
            num_variations: Number of variations

        Returns:
            List of (title, description_style) tuples
        """
        variations = []

        for i in range(num_variations):
            title = self.generate_title(mood, style, use_case, variation=i)

            # Different description styles
            desc_styles = ["detailed", "minimal", "emoji-heavy"]
            desc_style = desc_styles[i % len(desc_styles)]

            variations.append((title, desc_style))

        return variations


class PlaylistOrganizer:
    """Organize tracks into playlists by mood, season, or theme."""

    def __init__(self):
        """Initialize playlist organizer."""
        self.playlists = {}

    def create_playlist(self, name: str, description: str, tracks: List[str]) -> Dict:
        """Create a playlist."""
        playlist = {
            "name": name,
            "description": description,
            "tracks": tracks,
            "created_at": datetime.now().isoformat(),
            "track_count": len(tracks),
        }
        self.playlists[name] = playlist
        return playlist

    def organize_by_mood(self, tracks_metadata: List[TrackMetadata]) -> Dict[str, List[str]]:
        """Organize tracks into mood-based playlists."""
        mood_playlists = {}

        for metadata in tracks_metadata:
            mood = metadata.mood
            if mood not in mood_playlists:
                mood_playlists[mood] = []
            mood_playlists[mood].append(metadata.title)

        return mood_playlists

    def organize_by_season(self, tracks_metadata: List[TrackMetadata]) -> Dict[str, List[str]]:
        """Organize tracks into seasonal playlists."""
        # Determine season based on current date or metadata
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:
            season = "winter"
        elif current_month in [3, 4, 5]:
            season = "spring"
        elif current_month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

        seasonal_tracks = [m.title for m in tracks_metadata]
        return {season: seasonal_tracks}

    def create_series_playlists(
        self, base_name: str, tracks_per_playlist: int, all_tracks: List[str]
    ) -> List[Dict]:
        """Create series of playlists (e.g., "30 Days of Study Beats")."""
        series = []
        num_playlists = (len(all_tracks) + tracks_per_playlist - 1) // tracks_per_playlist

        for i in range(num_playlists):
            start_idx = i * tracks_per_playlist
            end_idx = min((i + 1) * tracks_per_playlist, len(all_tracks))
            tracks = all_tracks[start_idx:end_idx]

            playlist_name = f"{base_name} - Day {i+1}"
            playlist = self.create_playlist(
                name=playlist_name,
                description=f"Part {i+1} of {num_playlists} in the {base_name} series",
                tracks=tracks,
            )
            series.append(playlist)

        return series


# Example usage
if __name__ == "__main__":
    print("=== Metadata Generator ===\n")

    generator = MetadataGenerator()

    # Generate metadata for different use cases
    print("1. Study Track:")
    study_metadata = generator.generate_complete_metadata(
        mood="peaceful", style="lofi", use_case="study", bpm=75, key="Am", duration=3600
    )
    print(f"Title: {study_metadata.title}")
    print(f"Tags (first 5): {study_metadata.tags[:5]}")
    print(f"Description preview: {study_metadata.description[:200]}...")
    print()

    print("2. Sleep Track:")
    sleep_metadata = generator.generate_complete_metadata(
        mood="dreamy",
        style="ambient",
        use_case="sleep",
        bpm=60,
        key="C",
        duration=7200,
        seasonal="winter",
    )
    print(f"Title: {sleep_metadata.title}")
    print(f"Tags (first 5): {sleep_metadata.tags[:5]}")
    print()

    print("3. A/B Test Variations:")
    variations = generator.generate_ab_test_variations("chill", "lofi", "study", num_variations=3)
    for i, (title, style) in enumerate(variations, 1):
        print(f"  Variation {i}: {title} (Style: {style})")
    print()

    print("=== Playlist Organizer ===\n")
    organizer = PlaylistOrganizer()

    # Create sample tracks
    sample_tracks = []
    for mood in ["peaceful", "melancholic", "uplifting"]:
        for i in range(3):
            metadata = generator.generate_complete_metadata(
                mood=mood, style="lofi", use_case="study"
            )
            sample_tracks.append(metadata)

    # Organize by mood
    mood_playlists = organizer.organize_by_mood(sample_tracks)
    print("Mood-based playlists:")
    for mood, tracks in mood_playlists.items():
        print(f"  {mood.title()}: {len(tracks)} tracks")

    # Create series
    all_titles = [m.title for m in sample_tracks]
    series = organizer.create_series_playlists("30 Days of Study Beats", 3, all_titles)
    print(f"\nSeries created: {len(series)} playlists")
    for playlist in series:
        print(f"  {playlist['name']}: {playlist['track_count']} tracks")
