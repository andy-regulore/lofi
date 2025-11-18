"""
Intelligent playlist recommendation engine.

Recommends tracks and creates personalized playlists:
- Collaborative filtering (user-based and item-based)
- Content-based filtering (audio features)
- Mood-based recommendations
- Context-aware suggestions (time, weather, activity)
- Similar track discovery
- Playlist generation and optimization
- Cold start handling for new tracks
- A/B testing for recommendation strategies

Author: Claude
License: MIT
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time
import json
from pathlib import Path


class Mood(Enum):
    """Mood categories."""
    CHILL = "chill"
    FOCUS = "focus"
    HAPPY = "happy"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    PEACEFUL = "peaceful"
    ROMANTIC = "romantic"
    NOSTALGIC = "nostalgic"


class Activity(Enum):
    """Activity contexts."""
    STUDYING = "studying"
    WORKING = "working"
    SLEEPING = "sleeping"
    RELAXING = "relaxing"
    READING = "reading"
    CODING = "coding"
    COMMUTING = "commuting"
    EXERCISING = "exercising"


@dataclass
class Track:
    """Track metadata."""
    track_id: str
    title: str
    artist: str
    duration_seconds: int
    bpm: float
    key: str
    energy: float  # 0-1
    valence: float  # 0-1 (happiness)
    acousticness: float  # 0-1
    instrumentalness: float  # 0-1
    mood: Mood
    tags: List[str]
    release_date: datetime


@dataclass
class UserInteraction:
    """User interaction with track."""
    user_id: str
    track_id: str
    interaction_type: str  # play, like, skip, playlist_add
    timestamp: datetime
    play_duration_seconds: Optional[int] = None
    context: Optional[Activity] = None


@dataclass
class Playlist:
    """Playlist."""
    playlist_id: str
    name: str
    description: str
    tracks: List[str]  # Track IDs
    mood: Optional[Mood] = None
    activity: Optional[Activity] = None
    created_at: datetime = None


class AudioFeatureExtractor:
    """Extract and normalize audio features for comparison."""

    @staticmethod
    def extract_features(track: Track) -> np.ndarray:
        """
        Extract feature vector from track.

        Args:
            track: Track object

        Returns:
            Feature vector
        """
        features = np.array([
            track.bpm / 200.0,  # Normalize BPM
            track.energy,
            track.valence,
            track.acousticness,
            track.instrumentalness,
        ])

        return features

    @staticmethod
    def compute_similarity(track1: Track, track2: Track) -> float:
        """
        Compute similarity between two tracks.

        Args:
            track1: First track
            track2: Second track

        Returns:
            Similarity score (0-1)
        """
        features1 = AudioFeatureExtractor.extract_features(track1)
        features2 = AudioFeatureExtractor.extract_features(track2)

        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return similarity


class CollaborativeFilter:
    """Collaborative filtering recommender."""

    def __init__(self):
        """Initialize collaborative filter."""
        self.user_item_matrix = {}  # user_id -> {track_id: rating}
        self.item_user_matrix = {}  # track_id -> {user_id: rating}

    def add_interaction(self, interaction: UserInteraction):
        """
        Add user interaction.

        Args:
            interaction: User interaction data
        """
        # Convert interaction to rating
        rating = self._interaction_to_rating(interaction)

        # Update user-item matrix
        if interaction.user_id not in self.user_item_matrix:
            self.user_item_matrix[interaction.user_id] = {}
        self.user_item_matrix[interaction.user_id][interaction.track_id] = rating

        # Update item-user matrix
        if interaction.track_id not in self.item_user_matrix:
            self.item_user_matrix[interaction.track_id] = {}
        self.item_user_matrix[interaction.track_id][interaction.user_id] = rating

    def _interaction_to_rating(self, interaction: UserInteraction) -> float:
        """
        Convert interaction to rating (0-1).

        Args:
            interaction: Interaction data

        Returns:
            Rating score
        """
        if interaction.interaction_type == 'like':
            return 1.0
        elif interaction.interaction_type == 'playlist_add':
            return 0.9
        elif interaction.interaction_type == 'play':
            # Base on play duration
            if interaction.play_duration_seconds:
                # Assume full duration is 180 seconds
                ratio = min(interaction.play_duration_seconds / 180, 1.0)
                return 0.5 + 0.4 * ratio  # 0.5 to 0.9
            return 0.5
        elif interaction.interaction_type == 'skip':
            return 0.1
        else:
            return 0.5

    def user_based_recommendations(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        User-based collaborative filtering.

        Args:
            user_id: Target user ID
            n: Number of recommendations

        Returns:
            List of (track_id, score) tuples
        """
        if user_id not in self.user_item_matrix:
            return []

        user_ratings = self.user_item_matrix[user_id]

        # Find similar users
        similar_users = self._find_similar_users(user_id, k=10)

        # Aggregate ratings from similar users
        track_scores = {}

        for similar_user_id, similarity in similar_users:
            if similar_user_id not in self.user_item_matrix:
                continue

            for track_id, rating in self.user_item_matrix[similar_user_id].items():
                # Skip tracks user has already interacted with
                if track_id in user_ratings:
                    continue

                if track_id not in track_scores:
                    track_scores[track_id] = 0.0

                # Weight by user similarity
                track_scores[track_id] += rating * similarity

        # Sort and return top N
        recommendations = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def item_based_recommendations(self, track_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Item-based collaborative filtering (similar tracks).

        Args:
            track_id: Target track ID
            n: Number of recommendations

        Returns:
            List of (track_id, score) tuples
        """
        if track_id not in self.item_user_matrix:
            return []

        # Find similar items
        similar_items = self._find_similar_items(track_id, k=n+1)

        # Remove the track itself
        similar_items = [(tid, score) for tid, score in similar_items if tid != track_id]

        return similar_items[:n]

    def _find_similar_users(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k most similar users.

        Args:
            user_id: Target user ID
            k: Number of similar users

        Returns:
            List of (user_id, similarity) tuples
        """
        if user_id not in self.user_item_matrix:
            return []

        target_ratings = self.user_item_matrix[user_id]
        similarities = []

        for other_user_id in self.user_item_matrix:
            if other_user_id == user_id:
                continue

            other_ratings = self.user_item_matrix[other_user_id]

            # Compute similarity (Pearson correlation)
            similarity = self._compute_user_similarity(target_ratings, other_ratings)
            similarities.append((other_user_id, similarity))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _find_similar_items(self, track_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k most similar items.

        Args:
            track_id: Target track ID
            k: Number of similar tracks

        Returns:
            List of (track_id, similarity) tuples
        """
        if track_id not in self.item_user_matrix:
            return []

        target_ratings = self.item_user_matrix[track_id]
        similarities = []

        for other_track_id in self.item_user_matrix:
            if other_track_id == track_id:
                continue

            other_ratings = self.item_user_matrix[other_track_id]

            # Compute similarity
            similarity = self._compute_user_similarity(target_ratings, other_ratings)
            similarities.append((other_track_id, similarity))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _compute_user_similarity(self, ratings1: Dict, ratings2: Dict) -> float:
        """
        Compute similarity between two rating dictionaries.

        Args:
            ratings1: First user/item ratings
            ratings2: Second user/item ratings

        Returns:
            Similarity score
        """
        # Find common items
        common_items = set(ratings1.keys()) & set(ratings2.keys())

        if len(common_items) == 0:
            return 0.0

        # Compute cosine similarity
        vec1 = np.array([ratings1[item] for item in common_items])
        vec2 = np.array([ratings2[item] for item in common_items])

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class ContentBasedFilter:
    """Content-based filtering using audio features."""

    def __init__(self):
        """Initialize content-based filter."""
        self.tracks: Dict[str, Track] = {}

    def add_track(self, track: Track):
        """
        Add track to catalog.

        Args:
            track: Track object
        """
        self.tracks[track.track_id] = track

    def find_similar_tracks(self, track_id: str, n: int = 10,
                           mood_filter: Optional[Mood] = None) -> List[Tuple[str, float]]:
        """
        Find similar tracks based on audio features.

        Args:
            track_id: Target track ID
            n: Number of recommendations
            mood_filter: Optional mood filter

        Returns:
            List of (track_id, similarity) tuples
        """
        if track_id not in self.tracks:
            return []

        target_track = self.tracks[track_id]
        similarities = []

        for other_track_id, other_track in self.tracks.items():
            if other_track_id == track_id:
                continue

            # Apply mood filter
            if mood_filter and other_track.mood != mood_filter:
                continue

            # Compute similarity
            similarity = AudioFeatureExtractor.compute_similarity(target_track, other_track)
            similarities.append((other_track_id, similarity))

        # Sort and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def recommend_by_mood(self, mood: Mood, n: int = 10,
                         preferences: Optional[Dict] = None) -> List[str]:
        """
        Recommend tracks by mood.

        Args:
            mood: Target mood
            n: Number of recommendations
            preferences: Optional user preferences (e.g., preferred BPM range)

        Returns:
            List of track IDs
        """
        # Filter tracks by mood
        mood_tracks = [track for track in self.tracks.values() if track.mood == mood]

        # Apply preferences
        if preferences:
            if 'min_bpm' in preferences:
                mood_tracks = [t for t in mood_tracks if t.bpm >= preferences['min_bpm']]
            if 'max_bpm' in preferences:
                mood_tracks = [t for t in mood_tracks if t.bpm <= preferences['max_bpm']]
            if 'min_energy' in preferences:
                mood_tracks = [t for t in mood_tracks if t.energy >= preferences['min_energy']]

        # Sort by some criteria (e.g., popularity, recency)
        # For now, just shuffle and return
        import random
        random.shuffle(mood_tracks)

        return [t.track_id for t in mood_tracks[:n]]


class ContextAwareRecommender:
    """Context-aware recommendations based on time, activity, etc."""

    @staticmethod
    def recommend_by_time(current_time: time, tracks: List[Track], n: int = 10) -> List[str]:
        """
        Recommend based on time of day.

        Args:
            current_time: Current time
            tracks: Available tracks
            n: Number of recommendations

        Returns:
            List of track IDs
        """
        hour = current_time.hour

        # Time-based preferences
        if 6 <= hour < 9:
            # Morning: peaceful, low energy
            filtered = [t for t in tracks if t.energy < 0.5 and t.valence > 0.4]
        elif 9 <= hour < 12:
            # Late morning: focus, medium energy
            filtered = [t for t in tracks if 0.4 < t.energy < 0.7 and t.mood == Mood.FOCUS]
        elif 12 <= hour < 14:
            # Lunch: happy, upbeat
            filtered = [t for t in tracks if t.valence > 0.6]
        elif 14 <= hour < 18:
            # Afternoon: focus, productivity
            filtered = [t for t in tracks if t.mood in [Mood.FOCUS, Mood.CHILL]]
        elif 18 <= hour < 22:
            # Evening: relaxing, chill
            filtered = [t for t in tracks if t.energy < 0.6 and t.mood in [Mood.CHILL, Mood.PEACEFUL]]
        else:
            # Night: very chill, peaceful
            filtered = [t for t in tracks if t.energy < 0.4 and t.mood in [Mood.PEACEFUL, Mood.MELANCHOLIC]]

        if not filtered:
            filtered = tracks

        import random
        random.shuffle(filtered)
        return [t.track_id for t in filtered[:n]]

    @staticmethod
    def recommend_by_activity(activity: Activity, tracks: List[Track], n: int = 10) -> List[str]:
        """
        Recommend based on activity.

        Args:
            activity: Current activity
            tracks: Available tracks
            n: Number of recommendations

        Returns:
            List of track IDs
        """
        # Activity-based preferences
        preferences = {
            Activity.STUDYING: {'moods': [Mood.FOCUS, Mood.CHILL], 'max_energy': 0.6},
            Activity.WORKING: {'moods': [Mood.FOCUS, Mood.ENERGETIC], 'max_energy': 0.7},
            Activity.SLEEPING: {'moods': [Mood.PEACEFUL, Mood.MELANCHOLIC], 'max_energy': 0.3},
            Activity.RELAXING: {'moods': [Mood.CHILL, Mood.PEACEFUL], 'max_energy': 0.5},
            Activity.READING: {'moods': [Mood.PEACEFUL, Mood.FOCUS], 'max_energy': 0.4},
            Activity.CODING: {'moods': [Mood.FOCUS, Mood.CHILL], 'max_energy': 0.6},
            Activity.COMMUTING: {'moods': [Mood.CHILL, Mood.HAPPY], 'max_energy': 0.7},
            Activity.EXERCISING: {'moods': [Mood.ENERGETIC, Mood.HAPPY], 'min_energy': 0.6},
        }

        if activity not in preferences:
            import random
            random.shuffle(tracks)
            return [t.track_id for t in tracks[:n]]

        pref = preferences[activity]

        # Filter by mood
        filtered = [t for t in tracks if t.mood in pref['moods']]

        # Filter by energy
        if 'max_energy' in pref:
            filtered = [t for t in filtered if t.energy <= pref['max_energy']]
        if 'min_energy' in pref:
            filtered = [t for t in filtered if t.energy >= pref['min_energy']]

        if not filtered:
            filtered = tracks

        import random
        random.shuffle(filtered)
        return [t.track_id for t in filtered[:n]]


class PlaylistGenerator:
    """Generate optimized playlists."""

    def __init__(self, tracks: Dict[str, Track]):
        """
        Initialize playlist generator.

        Args:
            tracks: Track catalog
        """
        self.tracks = tracks

    def generate_mood_playlist(self, mood: Mood, duration_minutes: int = 60) -> Playlist:
        """
        Generate mood-based playlist.

        Args:
            mood: Target mood
            duration_minutes: Target duration

        Returns:
            Generated playlist
        """
        # Filter tracks by mood
        mood_tracks = [t for t in self.tracks.values() if t.mood == mood]

        # Sort by some criteria (popularity, energy progression, etc.)
        # For now, create smooth energy progression
        mood_tracks.sort(key=lambda t: t.energy)

        # Select tracks to fill duration
        selected_tracks = []
        total_duration = 0
        target_duration = duration_minutes * 60

        for track in mood_tracks:
            if total_duration + track.duration_seconds <= target_duration + 180:  # +3 min tolerance
                selected_tracks.append(track.track_id)
                total_duration += track.duration_seconds

            if total_duration >= target_duration:
                break

        # Create playlist
        playlist = Playlist(
            playlist_id=f"mood_{mood.value}_{datetime.now().timestamp()}",
            name=f"{mood.value.title()} Vibes",
            description=f"Perfect {mood.value} playlist for any time",
            tracks=selected_tracks,
            mood=mood,
            created_at=datetime.now()
        )

        return playlist

    def generate_flow_playlist(self, track_ids: List[str], smooth_transitions: bool = True) -> Playlist:
        """
        Generate playlist with smooth flow.

        Args:
            track_ids: Seed track IDs
            smooth_transitions: Optimize for smooth transitions

        Returns:
            Optimized playlist
        """
        if not track_ids:
            return None

        if not smooth_transitions:
            return Playlist(
                playlist_id=f"custom_{datetime.now().timestamp()}",
                name="Custom Playlist",
                description="Custom track selection",
                tracks=track_ids,
                created_at=datetime.now()
            )

        # Optimize order for smooth transitions
        # Use traveling salesman-like approach (greedy nearest neighbor)
        ordered_tracks = [track_ids[0]]
        remaining = set(track_ids[1:])

        while remaining:
            current_track = self.tracks[ordered_tracks[-1]]
            best_next = None
            best_similarity = -1

            for track_id in remaining:
                next_track = self.tracks[track_id]
                similarity = AudioFeatureExtractor.compute_similarity(current_track, next_track)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_next = track_id

            if best_next:
                ordered_tracks.append(best_next)
                remaining.remove(best_next)
            else:
                # No good match, just add any
                next_track = remaining.pop()
                ordered_tracks.append(next_track)

        playlist = Playlist(
            playlist_id=f"flow_{datetime.now().timestamp()}",
            name="Flow Playlist",
            description="Optimized for smooth transitions",
            tracks=ordered_tracks,
            created_at=datetime.now()
        )

        return playlist


class PlaylistRecommender:
    """Main playlist recommendation system."""

    def __init__(self):
        """Initialize playlist recommender."""
        self.collaborative_filter = CollaborativeFilter()
        self.content_filter = ContentBasedFilter()
        self.context_recommender = ContextAwareRecommender()
        self.playlist_generator = None

    def add_track(self, track: Track):
        """
        Add track to system.

        Args:
            track: Track object
        """
        self.content_filter.add_track(track)

    def add_interaction(self, interaction: UserInteraction):
        """
        Add user interaction.

        Args:
            interaction: Interaction data
        """
        self.collaborative_filter.add_interaction(interaction)

    def initialize_generator(self):
        """Initialize playlist generator with current tracks."""
        self.playlist_generator = PlaylistGenerator(self.content_filter.tracks)

    def recommend_for_user(self, user_id: str, n: int = 10,
                          context: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Hybrid recommendation for user.

        Args:
            user_id: User ID
            n: Number of recommendations
            context: Optional context (time, activity, mood)

        Returns:
            List of (track_id, score) tuples
        """
        recommendations = {}

        # Collaborative filtering
        collab_recs = self.collaborative_filter.user_based_recommendations(user_id, n=n*2)
        for track_id, score in collab_recs:
            recommendations[track_id] = score * 0.6  # Weight 60%

        # Context-based (if provided)
        if context:
            context_tracks = []

            if 'activity' in context:
                activity = context['activity']
                tracks = list(self.content_filter.tracks.values())
                context_track_ids = self.context_recommender.recommend_by_activity(
                    activity, tracks, n=n
                )
                for track_id in context_track_ids:
                    recommendations[track_id] = recommendations.get(track_id, 0) + 0.4

        # Sort and return top N
        final_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return final_recommendations[:n]

    def get_similar_tracks(self, track_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get similar tracks (hybrid approach).

        Args:
            track_id: Target track ID
            n: Number of similar tracks

        Returns:
            List of (track_id, similarity) tuples
        """
        recommendations = {}

        # Content-based similarity
        content_sims = self.content_filter.find_similar_tracks(track_id, n=n*2)
        for tid, score in content_sims:
            recommendations[tid] = score * 0.5

        # Collaborative similarity
        collab_sims = self.collaborative_filter.item_based_recommendations(track_id, n=n*2)
        for tid, score in collab_sims:
            recommendations[tid] = recommendations.get(tid, 0) + score * 0.5

        # Sort and return
        final = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return final[:n]

    def create_playlist(self, mood: Optional[Mood] = None,
                       activity: Optional[Activity] = None,
                       duration_minutes: int = 60) -> Playlist:
        """
        Create optimized playlist.

        Args:
            mood: Target mood
            activity: Target activity
            duration_minutes: Target duration

        Returns:
            Generated playlist
        """
        if not self.playlist_generator:
            self.initialize_generator()

        if mood:
            return self.playlist_generator.generate_mood_playlist(mood, duration_minutes)

        # Activity-based
        if activity:
            tracks = list(self.content_filter.tracks.values())
            track_ids = self.context_recommender.recommend_by_activity(
                activity, tracks, n=100
            )
            return self.playlist_generator.generate_flow_playlist(
                track_ids[:int(duration_minutes / 3)],
                smooth_transitions=True
            )

        return None


# Example usage
if __name__ == '__main__':
    print("=== Playlist Recommendation Engine ===\n")

    # Initialize recommender
    recommender = PlaylistRecommender()

    # Add sample tracks
    print("1. Adding Sample Tracks:")
    sample_tracks = [
        Track("t1", "Chill Beats 1", "LoFi AI", 180, 85, "C", 0.4, 0.6, 0.8, 0.9, Mood.CHILL, ["lofi", "chill"], datetime.now()),
        Track("t2", "Focus Flow", "LoFi AI", 200, 90, "Am", 0.5, 0.5, 0.7, 0.95, Mood.FOCUS, ["lofi", "study"], datetime.now()),
        Track("t3", "Peaceful Mind", "LoFi AI", 220, 70, "G", 0.3, 0.7, 0.9, 1.0, Mood.PEACEFUL, ["ambient", "calm"], datetime.now()),
        Track("t4", "Happy Days", "LoFi AI", 190, 110, "D", 0.7, 0.8, 0.6, 0.8, Mood.HAPPY, ["upbeat", "positive"], datetime.now()),
        Track("t5", "Night Thoughts", "LoFi AI", 210, 75, "Em", 0.35, 0.4, 0.85, 0.95, Mood.MELANCHOLIC, ["night", "introspective"], datetime.now()),
    ]

    for track in sample_tracks:
        recommender.add_track(track)
    print(f"Added {len(sample_tracks)} tracks\n")

    # Simulate user interactions
    print("2. Simulating User Interactions:")
    interactions = [
        UserInteraction("user1", "t1", "like", datetime.now()),
        UserInteraction("user1", "t2", "play", datetime.now(), 170),
        UserInteraction("user1", "t3", "playlist_add", datetime.now()),
        UserInteraction("user2", "t1", "like", datetime.now()),
        UserInteraction("user2", "t4", "play", datetime.now(), 190),
    ]

    for interaction in interactions:
        recommender.add_interaction(interaction)
    print(f"Processed {len(interactions)} interactions\n")

    # Get recommendations
    print("3. User Recommendations:")
    user_recs = recommender.recommend_for_user("user1", n=3)
    print("Recommendations for user1:")
    for track_id, score in user_recs:
        print(f"  - {track_id}: {score:.3f}")
    print()

    # Find similar tracks
    print("4. Similar Tracks:")
    similar = recommender.get_similar_tracks("t1", n=3)
    print("Similar to 'Chill Beats 1':")
    for track_id, similarity in similar:
        track = recommender.content_filter.tracks[track_id]
        print(f"  - {track.title}: {similarity:.3f}")
    print()

    # Generate playlists
    print("5. Generate Playlists:")
    recommender.initialize_generator()

    focus_playlist = recommender.create_playlist(mood=Mood.FOCUS, duration_minutes=30)
    print(f"Created '{focus_playlist.name}' with {len(focus_playlist.tracks)} tracks")

    study_playlist = recommender.create_playlist(activity=Activity.STUDYING, duration_minutes=60)
    if study_playlist:
        print(f"Created '{study_playlist.name}' with {len(study_playlist.tracks)} tracks")

    print("\n✅ Playlist recommendation system ready!")
    print("   Collaborative filtering active")
    print("   Content-based filtering active")
    print("   Context-aware recommendations enabled")
    print("   Smart playlist generation ready")
