"""
Multi-Platform Music Distribution

Distributes music to Spotify, Apple Music, Amazon Music, and other streaming platforms
via DistroKid API and other distribution services.

Author: Claude
License: MIT
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json


class MusicDistributor:
    """
    Multi-platform music distributor supporting:
    - Spotify (via DistroKid)
    - Apple Music (via DistroKid)
    - Amazon Music (via DistroKid)
    - YouTube Music (automatic)
    - Bandcamp
    - SoundCloud
    """

    def __init__(self, config: dict):
        """
        Initialize music distributor.

        Args:
            config: Configuration dict with API keys
        """
        self.config = config
        self.distrokid_api_key = config.get('distrokid', {}).get('api_key')
        self.soundcloud_token = config.get('soundcloud', {}).get('access_token')
        self.bandcamp_user = config.get('bandcamp', {}).get('username')
        self.bandcamp_pass = config.get('bandcamp', {}).get('password')

    def distribute_to_all_platforms(
        self,
        track_info: Dict,
        metadata: Dict,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Distribute track to all specified platforms.

        Args:
            track_info: Track information (audio_path, title, etc.)
            metadata: Metadata (artist, album, genre, etc.)
            platforms: List of platforms or None for all

        Returns:
            Dict with distribution results per platform
        """
        if platforms is None:
            platforms = ['spotify', 'apple_music', 'amazon_music', 'soundcloud']

        results = {}

        # Distribute to streaming services via DistroKid
        if any(p in platforms for p in ['spotify', 'apple_music', 'amazon_music']):
            streaming_platforms = [p for p in platforms if p in ['spotify', 'apple_music', 'amazon_music']]
            results['streaming'] = self.distribute_via_distrokid(
                track_info,
                metadata,
                platforms=streaming_platforms
            )

        # SoundCloud
        if 'soundcloud' in platforms:
            results['soundcloud'] = self.upload_to_soundcloud(track_info, metadata)

        # Bandcamp
        if 'bandcamp' in platforms:
            results['bandcamp'] = self.upload_to_bandcamp(track_info, metadata)

        return results

    def distribute_via_distrokid(
        self,
        track_info: Dict,
        metadata: Dict,
        platforms: List[str] = None,
        release_date: Optional[str] = None
    ) -> Dict:
        """
        Distribute music via DistroKid API.

        Args:
            track_info: Track information
            metadata: Track metadata
            platforms: Target platforms
            release_date: Release date (YYYY-MM-DD) or None for immediate

        Returns:
            Distribution result with tracking info
        """
        if not self.distrokid_api_key:
            return {
                'status': 'error',
                'message': 'DistroKid API key not configured',
                'instructions': 'Add api_key to config.json under distrokid section'
            }

        # Prepare metadata in DistroKid format
        distrokid_metadata = self._format_distrokid_metadata(metadata, release_date)

        # Upload audio file
        upload_result = self._upload_to_distrokid(
            track_info['audio_path'],
            distrokid_metadata
        )

        if upload_result['status'] == 'success':
            # Monitor distribution progress
            distribution_id = upload_result['distribution_id']
            return {
                'status': 'success',
                'distribution_id': distribution_id,
                'platforms': platforms or ['spotify', 'apple_music', 'amazon_music'],
                'release_date': release_date or 'immediate',
                'tracking_url': f"https://distrokid.com/releases/{distribution_id}",
                'estimated_live_date': self._calculate_go_live_date(release_date)
            }
        else:
            return upload_result

    def _format_distrokid_metadata(self, metadata: Dict, release_date: Optional[str]) -> Dict:
        """Format metadata for DistroKid API."""
        return {
            'artist': metadata.get('artist_name', 'LoFi AI'),
            'title': metadata.get('title', 'Untitled Track'),
            'album': metadata.get('album', 'LoFi Beats Collection'),
            'genre': metadata.get('genre', 'Lo-Fi'),
            'subgenre': metadata.get('subgenre', 'Chillhop'),
            'release_date': release_date or datetime.now().strftime('%Y-%m-%d'),
            'language': metadata.get('language', 'English'),
            'copyright': metadata.get('copyright', f"© {datetime.now().year} LoFi AI"),
            'upc': metadata.get('upc'),  # Optional
            'isrc': metadata.get('isrc'),  # Optional
            'explicit': False,
            'instrumental': metadata.get('instrumental', True),
            'tags': metadata.get('tags', ['lofi', 'chill', 'study', 'beats']),
            'description': metadata.get('description', ''),
            'artwork_path': metadata.get('thumbnail_path')  # Album artwork
        }

    def _upload_to_distrokid(self, audio_path: str, metadata: Dict) -> Dict:
        """
        Upload track to DistroKid (placeholder - requires actual API).

        Note: DistroKid doesn't have a public API. This is a template.
        Actual implementation would need to use their partner API or email-based workflow.
        """
        # PLACEHOLDER IMPLEMENTATION
        # Real implementation would use DistroKid's partner API

        print("📀 Preparing to distribute to streaming platforms via DistroKid")
        print(f"   Track: {metadata['title']}")
        print(f"   Artist: {metadata['artist']}")
        print(f"   Release Date: {metadata['release_date']}")

        # Simulate API call
        return {
            'status': 'pending',
            'distribution_id': f"DK_{int(time.time())}",
            'message': 'Track queued for distribution. Actual DistroKid API integration required.',
            'manual_steps': [
                '1. Log into DistroKid.com',
                '2. Upload the audio file: ' + audio_path,
                '3. Fill in metadata from the generated metadata.json',
                '4. Select platforms: Spotify, Apple Music, Amazon Music',
                '5. Set release date and submit'
            ],
            'automation_note': 'For full automation, contact DistroKid for partner API access'
        }

    def upload_to_soundcloud(self, track_info: Dict, metadata: Dict) -> Dict:
        """
        Upload track to SoundCloud.

        Args:
            track_info: Track information
            metadata: Track metadata

        Returns:
            Upload result
        """
        if not self.soundcloud_token:
            return {
                'status': 'error',
                'message': 'SoundCloud access token not configured'
            }

        # SoundCloud API endpoint
        url = 'https://api.soundcloud.com/tracks'

        # Prepare upload data
        files = {
            'track[asset_data]': open(track_info['audio_path'], 'rb')
        }

        data = {
            'track[title]': metadata.get('title'),
            'track[description]': metadata.get('description'),
            'track[genre]': metadata.get('genre', 'Lo-Fi'),
            'track[tag_list]': ' '.join(metadata.get('tags', [])),
            'track[sharing]': 'public',
            'track[downloadable]': False,
            'track[license]': 'all-rights-reserved',
            'oauth_token': self.soundcloud_token
        }

        # Upload artwork if available
        if 'thumbnail_path' in metadata:
            files['track[artwork_data]'] = open(metadata['thumbnail_path'], 'rb')

        try:
            response = requests.post(url, data=data, files=files)

            if response.status_code == 201:
                track_data = response.json()
                return {
                    'status': 'success',
                    'platform': 'soundcloud',
                    'track_id': track_data['id'],
                    'url': track_data['permalink_url'],
                    'uploaded_at': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'platform': 'soundcloud',
                    'error': response.text
                }
        except Exception as e:
            return {
                'status': 'error',
                'platform': 'soundcloud',
                'error': str(e)
            }
        finally:
            # Close file handles
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()

    def upload_to_bandcamp(self, track_info: Dict, metadata: Dict) -> Dict:
        """
        Upload track to Bandcamp.

        Note: Bandcamp doesn't have a public API. This requires manual upload
        or browser automation (Selenium).

        Args:
            track_info: Track information
            metadata: Track metadata

        Returns:
            Upload instruction dict
        """
        return {
            'status': 'manual_required',
            'platform': 'bandcamp',
            'message': 'Bandcamp requires manual upload (no public API)',
            'instructions': [
                '1. Log into bandcamp.com',
                '2. Go to your artist page',
                '3. Click "Add track"',
                f'4. Upload: {track_info["audio_path"]}',
                f'5. Title: {metadata.get("title")}',
                f'6. Tags: {", ".join(metadata.get("tags", []))}',
                '7. Set price or make free',
                '8. Publish'
            ],
            'audio_path': track_info['audio_path'],
            'metadata_file': self._save_metadata_for_manual_upload(track_info, metadata, 'bandcamp')
        }

    def track_distribution_status(self, distribution_id: str) -> Dict:
        """
        Track status of distribution across platforms.

        Args:
            distribution_id: Distribution ID from initial upload

        Returns:
            Status dict with per-platform information
        """
        # Placeholder - would query DistroKid API
        return {
            'distribution_id': distribution_id,
            'status': 'processing',
            'platforms': {
                'spotify': {
                    'status': 'pending',
                    'estimated_live': '7-14 days',
                    'url': None
                },
                'apple_music': {
                    'status': 'pending',
                    'estimated_live': '7-14 days',
                    'url': None
                },
                'amazon_music': {
                    'status': 'pending',
                    'estimated_live': '7-14 days',
                    'url': None
                }
            }
        }

    def get_streaming_analytics(self, track_id: str, platform: str) -> Dict:
        """
        Get streaming analytics for a distributed track.

        Args:
            track_id: Track ID on platform
            platform: Platform name

        Returns:
            Analytics data
        """
        # Placeholder for analytics API integration
        return {
            'track_id': track_id,
            'platform': platform,
            'streams': 0,
            'listeners': 0,
            'revenue': 0.0,
            'message': 'Analytics API integration required'
        }

    def _calculate_go_live_date(self, release_date: Optional[str]) -> str:
        """Calculate when track will go live on platforms."""
        if release_date:
            return release_date
        else:
            # Immediate release typically takes 7-14 days
            live_date = datetime.now() + timedelta(days=10)
            return live_date.strftime('%Y-%m-%d')

    def _save_metadata_for_manual_upload(
        self,
        track_info: Dict,
        metadata: Dict,
        platform: str
    ) -> str:
        """Save metadata to JSON file for manual upload reference."""
        output_dir = Path('output/distribution')
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = output_dir / f"{platform}_{track_info.get('track_id', 'track')}_metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump({
                'platform': platform,
                'track_info': track_info,
                'metadata': metadata,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)

        return str(metadata_file)


class SpotifyPlaylistPitcher:
    """Automated Spotify playlist pitching for better placement."""

    def __init__(self, spotify_token: str):
        """Initialize with Spotify API token."""
        self.token = spotify_token
        self.base_url = 'https://api.spotify.com/v1'

    def find_relevant_playlists(
        self,
        genre: str = 'lofi',
        min_followers: int = 1000,
        limit: int = 50
    ) -> List[Dict]:
        """
        Find relevant playlists for pitching.

        Args:
            genre: Music genre
            min_followers: Minimum playlist followers
            limit: Max results

        Returns:
            List of playlist dicts
        """
        headers = {'Authorization': f'Bearer {self.token}'}

        # Search for playlists
        search_url = f"{self.base_url}/search"
        params = {
            'q': genre,
            'type': 'playlist',
            'limit': limit
        }

        response = requests.get(search_url, headers=headers, params=params)

        if response.status_code == 200:
            playlists = response.json()['playlists']['items']

            # Filter by followers
            relevant = [
                p for p in playlists
                if p['followers']['total'] >= min_followers
            ]

            return relevant
        else:
            return []

    def pitch_to_playlist(
        self,
        playlist_id: str,
        track_uri: str,
        curator_email: Optional[str] = None
    ) -> Dict:
        """
        Pitch track to playlist curator.

        Note: Requires curator contact information.

        Args:
            playlist_id: Spotify playlist ID
            track_uri: Spotify track URI
            curator_email: Curator email (if available)

        Returns:
            Pitch result
        """
        # This would typically be done via email or Spotify's submission form
        return {
            'status': 'manual_required',
            'playlist_id': playlist_id,
            'track_uri': track_uri,
            'message': 'Contact curator directly via Spotify or email',
            'curator_email': curator_email
        }


# Convenience functions

def quick_distribute(
    audio_path: str,
    metadata: Dict,
    platforms: List[str] = None,
    config_path: str = 'config.json'
) -> Dict:
    """
    Quick distribution to all platforms.

    Args:
        audio_path: Path to audio file
        metadata: Track metadata
        platforms: Platforms to distribute to
        config_path: Path to config file

    Returns:
        Distribution results
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Create distributor
    distributor = MusicDistributor(config)

    # Distribute
    track_info = {
        'audio_path': audio_path,
        'track_id': metadata.get('track_id', str(int(time.time())))
    }

    results = distributor.distribute_to_all_platforms(
        track_info,
        metadata,
        platforms=platforms
    )

    return results


if __name__ == '__main__':
    # Demo usage
    print("🎵 Music Distribution System")
    print("=" * 60)

    demo_metadata = {
        'title': 'Chill Study Beats',
        'artist_name': 'LoFi AI',
        'album': 'Late Night Sessions',
        'genre': 'Lo-Fi',
        'tags': ['lofi', 'chill', 'study', 'beats', 'instrumental'],
        'description': 'Relaxing lo-fi beats for studying and focus'
    }

    print("\nExample distribution:")
    print(json.dumps(demo_metadata, indent=2))
    print("\nPlatforms: Spotify, Apple Music, Amazon Music, SoundCloud")
    print("\nTo enable: Add API keys to config.json")
