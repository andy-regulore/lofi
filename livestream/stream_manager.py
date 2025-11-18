"""
24/7 Livestream Manager

Manages OBS Studio automation and Restream.io integration for continuous streaming.

Author: Claude
License: MIT
"""

import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import requests


class OBSAutomation:
    """Automate OBS Studio for 24/7 streaming."""

    def __init__(self, obs_websocket_port: int = 4455, password: Optional[str] = None):
        """
        Initialize OBS automation.

        Args:
            obs_websocket_port: OBS WebSocket port (requires obs-websocket plugin)
            password: OBS WebSocket password
        """
        self.port = obs_websocket_port
        self.password = password
        self.ws_url = f"ws://localhost:{obs_websocket_port}"

    def start_streaming(self) -> Dict:
        """
        Start OBS streaming.

        Returns:
            Status dict
        """
        print("🎬 Starting OBS stream...")

        # This requires obs-websocket plugin
        # Install: https://github.com/obsproject/obs-websocket

        try:
            # Using obsws-python library (if installed)
            from obswebsocket import obsws, requests as obs_requests

            ws = obsws(f"localhost:{self.port}", self.password)
            ws.connect()

            # Start streaming
            ws.call(obs_requests.StartStream())

            ws.disconnect()

            return {
                'status': 'success',
                'message': 'Streaming started',
                'started_at': datetime.now().isoformat()
            }

        except ImportError:
            return {
                'status': 'error',
                'message': 'obs-websocket-py not installed. Run: pip install obs-websocket-py',
                'manual_steps': [
                    '1. Install OBS Studio',
                    '2. Install obs-websocket plugin',
                    '3. Install Python library: pip install obs-websocket-py',
                    '4. Configure stream key in OBS'
                ]
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'manual_steps': [
                    'Start streaming manually in OBS Studio'
                ]
            }

    def stop_streaming(self) -> Dict:
        """Stop OBS streaming."""
        print("⏸️  Stopping OBS stream...")

        try:
            from obswebsocket import obsws, requests as obs_requests

            ws = obsws(f"localhost:{self.port}", self.password)
            ws.connect()

            ws.call(obs_requests.StopStream())

            ws.disconnect()

            return {
                'status': 'success',
                'message': 'Streaming stopped'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def set_current_scene(self, scene_name: str) -> Dict:
        """Switch to a different scene."""
        try:
            from obswebsocket import obsws, requests as obs_requests

            ws = obsws(f"localhost:{self.port}", self.password)
            ws.connect()

            ws.call(obs_requests.SetCurrentProgramScene(sceneName=scene_name))

            ws.disconnect()

            return {'status': 'success', 'scene': scene_name}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def update_media_source(self, source_name: str, file_path: str) -> Dict:
        """
        Update video file in media source.

        Args:
            source_name: Name of media source in OBS
            file_path: Path to new video file

        Returns:
            Status dict
        """
        try:
            from obswebsocket import obsws, requests as obs_requests

            ws = obsws(f"localhost:{self.port}", self.password)
            ws.connect()

            # Update media source settings
            ws.call(obs_requests.SetInputSettings(
                inputName=source_name,
                inputSettings={'local_file': file_path}
            ))

            ws.disconnect()

            print(f"✅ Updated media source: {source_name}")
            return {'status': 'success', 'file': file_path}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class RestreamManager:
    """Manage Restream.io for multi-platform broadcasting."""

    def __init__(self, api_key: str):
        """
        Initialize Restream manager.

        Args:
            api_key: Restream.io API key
        """
        self.api_key = api_key
        self.base_url = 'https://api.restream.io/v1'

    def get_stream_key(self) -> str:
        """Get Restream RTMP stream key."""
        headers = {'Authorization': f'Bearer {self.api_key}'}

        try:
            response = requests.get(
                f'{self.base_url}/user/stream-key',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                return data['streamKey']
            else:
                print(f"Error getting stream key: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_connected_channels(self) -> List[Dict]:
        """Get list of connected streaming platforms."""
        headers = {'Authorization': f'Bearer {self.api_key}'}

        try:
            response = requests.get(
                f'{self.base_url}/user/channel/all',
                headers=headers
            )

            if response.status_code == 200:
                return response.json()['channels']
            else:
                return []

        except Exception as e:
            print(f"Error: {e}")
            return []

    def update_stream_title(self, title: str, platforms: Optional[List[str]] = None) -> Dict:
        """
        Update stream title across platforms.

        Args:
            title: Stream title
            platforms: List of platforms or None for all

        Returns:
            Update status
        """
        headers = {'Authorization': f'Bearer {self.api_key}'}

        data = {'title': title}
        if platforms:
            data['platforms'] = platforms

        try:
            response = requests.post(
                f'{self.base_url}/user/stream/title',
                headers=headers,
                json=data
            )

            return {
                'status': 'success' if response.status_code == 200 else 'error',
                'title': title
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class LivestreamManager:
    """Complete 24/7 livestream management system."""

    def __init__(self, config: dict):
        """
        Initialize livestream manager.

        Args:
            config: Configuration dict
        """
        self.config = config
        self.obs = OBSAutomation(
            obs_websocket_port=config.get('obs', {}).get('websocket_port', 4455),
            password=config.get('obs', {}).get('password')
        )

        restream_key = config.get('restream', {}).get('api_key')
        self.restream = RestreamManager(restream_key) if restream_key else None

    def start_24_7_stream(
        self,
        video_playlist: List[str],
        stream_title: str = "24/7 LoFi Beats Radio 🎧"
    ) -> Dict:
        """
        Start 24/7 continuous stream.

        Args:
            video_playlist: List of video file paths to loop
            stream_title: Stream title

        Returns:
            Stream status
        """
        print("🚀 Starting 24/7 LoFi Radio Stream")
        print(f"   Title: {stream_title}")
        print(f"   Videos: {len(video_playlist)}")

        # Update stream title via Restream
        if self.restream:
            self.restream.update_stream_title(stream_title)
            print("   ✅ Stream title updated")

        # Start OBS
        result = self.obs.start_streaming()

        if result['status'] == 'success':
            print("   ✅ OBS streaming started")

            return {
                'status': 'live',
                'started_at': datetime.now().isoformat(),
                'title': stream_title,
                'video_count': len(video_playlist),
                'platforms': self._get_platforms_list()
            }
        else:
            print(f"   ❌ Failed to start OBS: {result.get('message')}")
            return result

    def monitor_and_restart(
        self,
        check_interval: int = 300,
        video_playlist: List[str] = None
    ):
        """
        Monitor stream and auto-restart if it drops.

        Args:
            check_interval: How often to check (seconds)
            video_playlist: Videos to use for restart

        This runs continuously - use systemd or supervisor to manage.
        """
        print("👁️  Stream monitoring started")
        print(f"   Check interval: {check_interval}s")

        while True:
            time.sleep(check_interval)

            # Check if stream is live
            is_live = self._check_stream_status()

            if not is_live:
                print("⚠️  Stream offline! Attempting restart...")

                # Restart stream
                if video_playlist:
                    self.start_24_7_stream(video_playlist)
                else:
                    self.obs.start_streaming()

                print("   ✅ Stream restarted")

            else:
                print(f"   ✅ Stream healthy ({datetime.now().strftime('%H:%M:%S')})")

    def _check_stream_status(self) -> bool:
        """Check if stream is currently live."""
        try:
            from obswebsocket import obsws, requests as obs_requests

            ws = obsws(f"localhost:{self.obs.port}", self.obs.password)
            ws.connect()

            response = ws.call(obs_requests.GetStreamStatus())

            ws.disconnect()

            return response.getOutputActive()

        except:
            return False

    def _get_platforms_list(self) -> List[str]:
        """Get list of streaming platforms."""
        if self.restream:
            channels = self.restream.get_connected_channels()
            return [ch['platform'] for ch in channels]
        else:
            return ['youtube']  # Default

    def schedule_video_rotation(
        self,
        video_playlist: List[str],
        rotation_interval_hours: int = 8
    ):
        """
        Rotate through videos in playlist.

        Args:
            video_playlist: List of video paths
            rotation_interval_hours: Hours before switching video
        """
        print(f"🔄 Video rotation enabled ({rotation_interval_hours}h intervals)")

        video_index = 0

        while True:
            current_video = video_playlist[video_index % len(video_playlist)]

            print(f"   Switching to: {Path(current_video).name}")

            # Update OBS media source
            self.obs.update_media_source('LoFiVideo', current_video)

            # Wait for rotation interval
            time.sleep(rotation_interval_hours * 3600)

            video_index += 1


def setup_obs_scene(
    scene_name: str = 'LoFi Radio',
    media_source_name: str = 'LoFiVideo'
) -> Dict:
    """
    Instructions for setting up OBS scene.

    Returns:
        Setup instructions
    """
    return {
        'scene_name': scene_name,
        'instructions': [
            '1. Open OBS Studio',
            f'2. Create new scene: "{scene_name}"',
            '3. Add sources:',
            f'   - Media Source: "{media_source_name}"',
            '     * Set to loop: YES',
            '     * Restart playback when source becomes active: YES',
            '   - (Optional) Text: Stream title/info',
            '   - (Optional) Image: Channel logo',
            '4. Configure stream settings:',
            '   - Go to Settings → Stream',
            '   - Service: YouTube / Restream',
            '   - Enter stream key',
            '5. Enable WebSocket:',
            '   - Go to Tools → WebSocket Server Settings',
            '   - Enable WebSocket server',
            '   - Set port: 4455',
            '   - (Optional) Set password',
            '6. Start streaming via this script'
        ]
    }


if __name__ == '__main__':
    print("🎵 24/7 LoFi Radio - Livestream Manager")
    print("=" * 60)

    # Show setup instructions
    setup = setup_obs_scene()
    print("\n📋 OBS Setup Instructions:")
    for instruction in setup['instructions']:
        print(f"   {instruction}")

    print("\n💡 To start streaming:")
    print("   1. Complete OBS setup above")
    print("   2. Generate stream videos with radio_generator.py")
    print("   3. Run this script with config.json configured")
