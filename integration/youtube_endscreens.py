"""
YouTube End Screens and Cards Automation

Automates the creation and management of YouTube end screens and cards
to improve viewer retention and channel growth.

Features:
- Automated end screen templates
- Dynamic card placement
- Subscribe button positioning
- Best-performing video promotion
- Playlist recommendations
- Channel branding elements

Author: Claude
License: MIT
"""

import json
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndScreenElement:
    """Represents a single end screen element."""

    def __init__(self, element_type: str, position: Dict[str, float],
                 duration: int = 15, **kwargs):
        """
        Initialize end screen element.

        Args:
            element_type: Type (video, playlist, subscribe, channel)
            position: Position dict with left, top, width, height (0-1 normalized)
            duration: Duration in seconds before video end
            **kwargs: Additional element-specific properties
        """
        self.type = element_type
        self.position = position
        self.duration = duration
        self.properties = kwargs

    def to_youtube_api_format(self) -> Dict:
        """Convert to YouTube API format."""
        element = {
            'type': self.type,
            'endTimeMs': self.duration * 1000,  # YouTube uses milliseconds from video end
            'startMs': 0,  # Start at beginning of end screen period
            **self.position
        }

        # Add type-specific properties
        if self.type == 'video':
            element['videoId'] = self.properties.get('video_id')
        elif self.type == 'playlist':
            element['playlistId'] = self.properties.get('playlist_id')

        return element


class EndScreenTemplate:
    """Predefined end screen templates."""

    @staticmethod
    def default_template() -> List[EndScreenElement]:
        """
        Default end screen layout:
        - Subscribe button (top-right)
        - Best video recommendation (center-left)
        - Latest upload (center-right)
        """
        return [
            # Subscribe button (top-right)
            EndScreenElement(
                element_type='subscribe',
                position={'left': 0.7, 'top': 0.1, 'width': 0.25, 'height': 0.15},
                duration=15
            ),
            # Best performing video (left)
            EndScreenElement(
                element_type='video',
                position={'left': 0.05, 'top': 0.35, 'width': 0.4, 'height': 0.5},
                duration=15,
                video_id='BEST_VIDEO'  # Placeholder - will be replaced dynamically
            ),
            # Latest upload (right)
            EndScreenElement(
                element_type='video',
                position={'left': 0.55, 'top': 0.35, 'width': 0.4, 'height': 0.5},
                duration=15,
                video_id='LATEST_VIDEO'  # Placeholder
            )
        ]

    @staticmethod
    def playlist_focus_template() -> List[EndScreenElement]:
        """
        Playlist-focused template:
        - Subscribe button
        - Main playlist
        - Related playlist
        """
        return [
            # Subscribe button (top-center)
            EndScreenElement(
                element_type='subscribe',
                position={'left': 0.375, 'top': 0.05, 'width': 0.25, 'height': 0.15},
                duration=15
            ),
            # Main playlist (large, center)
            EndScreenElement(
                element_type='playlist',
                position={'left': 0.1, 'top': 0.3, 'width': 0.8, 'height': 0.5},
                duration=15,
                playlist_id='MAIN_PLAYLIST'
            )
        ]

    @staticmethod
    def channel_growth_template() -> List[EndScreenElement]:
        """
        Channel growth template:
        - Large subscribe button
        - Channel link
        - Best video
        """
        return [
            # Large subscribe button (top, prominent)
            EndScreenElement(
                element_type='subscribe',
                position={'left': 0.25, 'top': 0.1, 'width': 0.5, 'height': 0.2},
                duration=15
            ),
            # Channel browse (left)
            EndScreenElement(
                element_type='channel',
                position={'left': 0.05, 'top': 0.45, 'width': 0.4, 'height': 0.4},
                duration=15
            ),
            # Best video (right)
            EndScreenElement(
                element_type='video',
                position={'left': 0.55, 'top': 0.45, 'width': 0.4, 'height': 0.4},
                duration=15,
                video_id='BEST_VIDEO'
            )
        ]


class Card:
    """Represents a YouTube card."""

    def __init__(self, card_type: str, timing_seconds: int, **kwargs):
        """
        Initialize card.

        Args:
            card_type: Card type (video, playlist, poll, link)
            timing_seconds: When to show card (seconds from start)
            **kwargs: Card-specific properties
        """
        self.type = card_type
        self.timing = timing_seconds
        self.properties = kwargs

    def to_youtube_api_format(self) -> Dict:
        """Convert to YouTube API format."""
        card = {
            'type': self.type,
            'timing': {
                'offsetMs': self.timing * 1000,
                'type': 'offsetFromStart'
            }
        }

        # Add type-specific properties
        if self.type == 'video':
            card['videoId'] = self.properties.get('video_id')
            card['message'] = self.properties.get('message', 'Check out this video!')
        elif self.type == 'playlist':
            card['playlistId'] = self.properties.get('playlist_id')
            card['message'] = self.properties.get('message', 'View playlist')
        elif self.type == 'poll':
            card['question'] = self.properties.get('question')
            card['options'] = self.properties.get('options', [])

        return card


class YouTubeEndScreenManager:
    """
    Manages YouTube end screens and cards automation.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize end screen manager.

        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key

    def create_end_screen(self, video_id: str, template: str = 'default',
                         custom_elements: Optional[List[EndScreenElement]] = None) -> Dict:
        """
        Create end screen for a video.

        Args:
            video_id: YouTube video ID
            template: Template name ('default', 'playlist_focus', 'channel_growth')
            custom_elements: Optional custom elements (overrides template)

        Returns:
            End screen configuration
        """
        # Get template elements
        if custom_elements:
            elements = custom_elements
        else:
            if template == 'playlist_focus':
                elements = EndScreenTemplate.playlist_focus_template()
            elif template == 'channel_growth':
                elements = EndScreenTemplate.channel_growth_template()
            else:
                elements = EndScreenTemplate.default_template()

        # Replace placeholders with actual IDs
        elements = self._replace_placeholders(elements, video_id)

        # Create end screen configuration
        end_screen_config = {
            'video_id': video_id,
            'elements': [e.to_youtube_api_format() for e in elements],
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"Created end screen for video {video_id} using {template} template")
        return end_screen_config

    def _replace_placeholders(self, elements: List[EndScreenElement],
                             current_video_id: str) -> List[EndScreenElement]:
        """
        Replace placeholder video/playlist IDs with actual IDs.

        Args:
            elements: List of end screen elements
            current_video_id: Current video ID

        Returns:
            Elements with replaced placeholders
        """
        # In production, this would:
        # 1. Query analytics for best performing video
        # 2. Get latest upload
        # 3. Get relevant playlists

        for element in elements:
            if element.properties.get('video_id') == 'BEST_VIDEO':
                # Replace with best performing video
                best_video = self._get_best_performing_video(exclude=current_video_id)
                if best_video:
                    element.properties['video_id'] = best_video

            elif element.properties.get('video_id') == 'LATEST_VIDEO':
                # Replace with latest upload
                latest_video = self._get_latest_video(exclude=current_video_id)
                if latest_video:
                    element.properties['video_id'] = latest_video

            elif element.properties.get('playlist_id') == 'MAIN_PLAYLIST':
                # Replace with relevant playlist
                playlist = self._get_relevant_playlist(current_video_id)
                if playlist:
                    element.properties['playlist_id'] = playlist

        return elements

    def _get_best_performing_video(self, exclude: Optional[str] = None) -> Optional[str]:
        """
        Get best performing video ID from channel.

        In production, this would query YouTube Analytics API.
        For now, returns a placeholder.
        """
        # Placeholder - would query analytics
        logger.debug("Getting best performing video")
        return None  # Return None to skip in template

    def _get_latest_video(self, exclude: Optional[str] = None) -> Optional[str]:
        """Get latest uploaded video ID."""
        # Placeholder - would query channel uploads
        logger.debug("Getting latest video")
        return None

    def _get_relevant_playlist(self, video_id: str) -> Optional[str]:
        """Get relevant playlist for video."""
        # Placeholder - would match video to playlist
        logger.debug(f"Getting relevant playlist for {video_id}")
        return None

    def add_cards(self, video_id: str, card_placements: List[Dict]) -> List[Card]:
        """
        Add cards to a video at strategic points.

        Args:
            video_id: YouTube video ID
            card_placements: List of card placement configs

        Returns:
            List of Card objects
        """
        cards = []

        for placement in card_placements:
            card = Card(
                card_type=placement['type'],
                timing_seconds=placement['timing'],
                **placement.get('properties', {})
            )
            cards.append(card)

        logger.info(f"Added {len(cards)} cards to video {video_id}")
        return cards

    def create_smart_cards(self, video_id: str, duration_seconds: int,
                          video_type: str = 'music') -> List[Card]:
        """
        Create strategically placed cards based on video type and duration.

        Args:
            video_id: YouTube video ID
            duration_seconds: Video duration
            video_type: Type of video ('music', 'tutorial', 'livestream')

        Returns:
            List of strategically placed cards
        """
        cards = []

        if video_type == 'music':
            # For music videos, place cards sparsely to avoid disruption
            # Card at 25% through
            if duration_seconds > 240:  # Only for videos > 4 minutes
                cards.append(Card(
                    card_type='playlist',
                    timing_seconds=int(duration_seconds * 0.25),
                    playlist_id='STUDY_MUSIC_PLAYLIST',
                    message='More study beats 📚'
                ))

            # Card at 60% through
            if duration_seconds > 600:  # Only for videos > 10 minutes
                cards.append(Card(
                    card_type='video',
                    timing_seconds=int(duration_seconds * 0.6),
                    video_id='SIMILAR_VIDEO',
                    message='You might also like this 🎵'
                ))

        elif video_type == 'tutorial':
            # For tutorials, more cards are acceptable
            # Early card (10% in)
            cards.append(Card(
                card_type='playlist',
                timing_seconds=int(duration_seconds * 0.1),
                playlist_id='TUTORIAL_PLAYLIST',
                message='Full tutorial series'
            ))

            # Mid-point card
            cards.append(Card(
                card_type='video',
                timing_seconds=int(duration_seconds * 0.5),
                video_id='RELATED_TUTORIAL',
                message='Related tutorial'
            ))

            # Late card with poll
            cards.append(Card(
                card_type='poll',
                timing_seconds=int(duration_seconds * 0.8),
                question='Was this tutorial helpful?',
                options=['Yes, very helpful!', 'Somewhat helpful', 'Not really']
            ))

        logger.info(f"Created {len(cards)} smart cards for {video_type} video")
        return cards

    def batch_apply_end_screens(self, video_ids: List[str], template: str = 'default') -> Dict:
        """
        Apply end screens to multiple videos at once.

        Args:
            video_ids: List of video IDs
            template: Template to use

        Returns:
            Application results
        """
        results = {
            'successful': [],
            'failed': []
        }

        for video_id in video_ids:
            try:
                config = self.create_end_screen(video_id, template)
                results['successful'].append({
                    'video_id': video_id,
                    'config': config
                })
            except Exception as e:
                logger.error(f"Failed to apply end screen to {video_id}: {e}")
                results['failed'].append({
                    'video_id': video_id,
                    'error': str(e)
                })

        logger.info(f"Applied end screens to {len(results['successful'])}/{len(video_ids)} videos")
        return results

    def save_template(self, name: str, elements: List[EndScreenElement],
                     description: str = "") -> str:
        """
        Save a custom end screen template.

        Args:
            name: Template name
            elements: List of end screen elements
            description: Optional description

        Returns:
            Template file path
        """
        template_dir = Path("data/end_screen_templates")
        template_dir.mkdir(parents=True, exist_ok=True)

        template_data = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'elements': [
                {
                    'type': e.type,
                    'position': e.position,
                    'duration': e.duration,
                    'properties': e.properties
                }
                for e in elements
            ]
        }

        template_path = template_dir / f"{name}.json"
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)

        logger.info(f"Saved template '{name}' to {template_path}")
        return str(template_path)

    def load_template(self, name: str) -> List[EndScreenElement]:
        """Load a saved template."""
        template_path = Path(f"data/end_screen_templates/{name}.json")

        if not template_path.exists():
            raise FileNotFoundError(f"Template '{name}' not found")

        with open(template_path, 'r') as f:
            template_data = json.load(f)

        elements = []
        for elem_data in template_data['elements']:
            element = EndScreenElement(
                element_type=elem_data['type'],
                position=elem_data['position'],
                duration=elem_data['duration'],
                **elem_data['properties']
            )
            elements.append(element)

        logger.info(f"Loaded template '{name}' with {len(elements)} elements")
        return elements


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = YouTubeEndScreenManager()

    # Create end screen for a video
    print("=== Creating End Screen ===")
    end_screen = manager.create_end_screen(
        video_id="dQw4w9WgXcQ",
        template="default"
    )

    print(f"Created end screen with {len(end_screen['elements'])} elements:")
    for i, element in enumerate(end_screen['elements'], 1):
        print(f"  {i}. {element['type']} at position {element['left']},{element['top']}")

    # Create smart cards
    print("\n=== Creating Smart Cards ===")
    cards = manager.create_smart_cards(
        video_id="dQw4w9WgXcQ",
        duration_seconds=3600,  # 1 hour video
        video_type="music"
    )

    print(f"Created {len(cards)} cards:")
    for i, card in enumerate(cards, 1):
        print(f"  {i}. {card.type} at {card.timing}s")

    # Create and save custom template
    print("\n=== Creating Custom Template ===")
    custom_elements = [
        EndScreenElement(
            element_type='subscribe',
            position={'left': 0.4, 'top': 0.1, 'width': 0.2, 'height': 0.15},
            duration=20
        ),
        EndScreenElement(
            element_type='playlist',
            position={'left': 0.1, 'top': 0.4, 'width': 0.8, 'height': 0.5},
            duration=20,
            playlist_id='CUSTOM_PLAYLIST'
        )
    ]

    template_path = manager.save_template(
        name="minimal_subscribe",
        elements=custom_elements,
        description="Minimal template focusing on subscription"
    )
    print(f"Saved custom template to: {template_path}")

    # Batch apply to multiple videos
    print("\n=== Batch Application ===")
    video_list = ["video1", "video2", "video3"]
    results = manager.batch_apply_end_screens(video_list, template="channel_growth")
    print(f"Successfully applied to {len(results['successful'])} videos")
