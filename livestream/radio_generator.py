"""
24/7 LoFi Radio Generator

Creates long-form video loops (8+ hours) for continuous livestreaming.
Combines multiple tracks with seamless transitions.

Author: Claude
License: MIT
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json
import random
from datetime import datetime, timedelta


class LoFiRadioGenerator:
    """Generates long-form video content for 24/7 streaming."""

    def __init__(self, output_dir: str = 'output/livestream'):
        """
        Initialize radio generator.

        Args:
            output_dir: Directory for livestream videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_stream_video(
        self,
        track_list: List[str],
        duration_hours: int = 8,
        output_name: Optional[str] = None,
        visual_template: str = 'classic_lofi'
    ) -> Dict:
        """
        Create long-form video for streaming.

        Args:
            track_list: List of audio file paths
            duration_hours: Target duration in hours
            output_name: Output video name (auto-generated if None)
            visual_template: Video template to use

        Returns:
            Dict with video info
        """
        print(f"\n🎥 Creating {duration_hours}-hour livestream video")
        print(f"   Tracks: {len(track_list)}")
        print(f"   Template: {visual_template}")

        # Generate output filename
        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f"lofi_radio_{duration_hours}h_{timestamp}.mp4"

        output_path = self.output_dir / output_name

        # Calculate how many loops needed
        target_duration = duration_hours * 3600  # seconds
        loops_needed = self._calculate_loops(track_list, target_duration)

        print(f"   Loops: {loops_needed}")

        # Create concatenation file list
        concat_file = self._create_concat_file(track_list, loops_needed)

        # Generate video with ffmpeg
        video_path = self._generate_long_video(
            concat_file,
            output_path,
            visual_template,
            target_duration
        )

        return {
            'video_path': str(video_path),
            'duration_hours': duration_hours,
            'track_count': len(track_list),
            'loops': loops_needed,
            'created_at': datetime.now().isoformat(),
            'file_size_gb': video_path.stat().st_size / (1024**3) if video_path.exists() else 0
        }

    def create_continuous_stream(
        self,
        track_directory: str,
        hours_per_video: int = 8,
        videos_to_generate: int = 3,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Create multiple videos for continuous 24/7 streaming.

        Args:
            track_directory: Directory containing audio tracks
            hours_per_video: Hours per video file
            videos_to_generate: Number of videos to create
            shuffle: Randomize track order

        Returns:
            List of generated video dicts
        """
        print(f"\n🔄 Creating continuous stream content")
        print(f"   Videos: {videos_to_generate} x {hours_per_video}h")

        # Get all audio tracks
        track_dir = Path(track_directory)
        audio_files = list(track_dir.glob('*.wav')) + list(track_dir.glob('*.mp3'))

        if not audio_files:
            print(f"   ⚠️  No audio files found in {track_directory}")
            return []

        print(f"   Available tracks: {len(audio_files)}")

        videos = []

        for i in range(videos_to_generate):
            # Shuffle for variety
            if shuffle:
                track_list = random.sample(audio_files, min(len(audio_files), 50))
            else:
                track_list = audio_files

            # Create video
            video_info = self.create_stream_video(
                [str(t) for t in track_list],
                duration_hours=hours_per_video,
                output_name=f"lofi_radio_part{i+1:02d}.mp4"
            )

            videos.append(video_info)

            print(f"   ✅ Part {i+1}/{videos_to_generate} complete")

        return videos

    def _calculate_loops(self, track_list: List[str], target_duration: int) -> int:
        """Calculate how many times to loop playlist."""
        # Estimate average track duration (assume 3 minutes if can't determine)
        total_playlist_duration = len(track_list) * 180

        if total_playlist_duration == 0:
            return 1

        loops = int(target_duration / total_playlist_duration) + 1
        return max(1, loops)

    def _create_concat_file(self, track_list: List[str], loops: int) -> Path:
        """Create ffmpeg concatenation file."""
        concat_file = self.output_dir / 'concat_list.txt'

        with open(concat_file, 'w') as f:
            for _ in range(loops):
                for track in track_list:
                    # ffmpeg concat format
                    f.write(f"file '{Path(track).absolute()}'\n")

        return concat_file

    def _generate_long_video(
        self,
        concat_file: Path,
        output_path: Path,
        visual_template: str,
        target_duration: int
    ) -> Path:
        """Generate long video using ffmpeg."""

        # For simplicity, we'll create a video with a static image and audio
        # In production, you'd use your existing video generation system

        print("   Rendering video (this may take a while)...")

        # Create simple visualization (static background with audio)
        # You can replace this with your video_generator.py templates

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-loop', '1',
            '-i', 'static/default_background.png',  # Default background
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-tune', 'stillimage',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-t', str(target_duration),
            str(output_path)
        ]

        # Run ffmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"   ✅ Video created: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error creating video: {e.stderr.decode()}")
            # Fallback: just concatenate audio
            audio_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path.with_suffix('.mp3'))
            ]
            subprocess.run(audio_cmd, check=True)
            output_path = output_path.with_suffix('.mp3')
            print(f"   ✅ Audio loop created: {output_path}")

        return output_path

    def create_stream_schedule(
        self,
        num_days: int = 7,
        hours_per_video: int = 8
    ) -> List[Dict]:
        """
        Create a weekly streaming schedule.

        Args:
            num_days: Number of days to schedule
            hours_per_video: Hours per video segment

        Returns:
            List of schedule entries
        """
        schedule = []
        videos_per_day = 24 // hours_per_video

        for day in range(num_days):
            for slot in range(videos_per_day):
                start_time = datetime.now() + timedelta(
                    days=day,
                    hours=slot * hours_per_video
                )

                schedule.append({
                    'day': day + 1,
                    'slot': slot + 1,
                    'start_time': start_time.isoformat(),
                    'duration_hours': hours_per_video,
                    'video_file': f'lofi_radio_day{day+1}_slot{slot+1}.mp4',
                    'status': 'pending'
                })

        return schedule

    def save_schedule(self, schedule: List[Dict], filename: str = 'stream_schedule.json'):
        """Save streaming schedule to JSON."""
        schedule_file = self.output_dir / filename

        with open(schedule_file, 'w') as f:
            json.dump({
                'created_at': datetime.now().isoformat(),
                'total_videos': len(schedule),
                'schedule': schedule
            }, f, indent=2)

        print(f"✅ Schedule saved: {schedule_file}")
        return schedule_file


def quick_create_stream(
    track_directory: str = 'output/audio',
    hours: int = 8
) -> str:
    """
    Quick function to create a streaming video.

    Args:
        track_directory: Directory with audio tracks
        hours: Duration in hours

    Returns:
        Path to generated video
    """
    generator = LoFiRadioGenerator()

    # Get tracks
    tracks = list(Path(track_directory).glob('*.wav'))

    if not tracks:
        print("❌ No tracks found in", track_directory)
        return None

    # Create video
    result = generator.create_stream_video(
        [str(t) for t in tracks],
        duration_hours=hours
    )

    return result['video_path']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate 24/7 LoFi radio stream videos')
    parser.add_argument('--tracks', default='output/audio', help='Track directory')
    parser.add_argument('--hours', type=int, default=8, help='Hours per video')
    parser.add_argument('--count', type=int, default=1, help='Number of videos')

    args = parser.parse_args()

    generator = LoFiRadioGenerator()

    if args.count == 1:
        # Single video
        video_path = quick_create_stream(args.tracks, args.hours)
        print(f"\n✅ Stream video ready: {video_path}")
    else:
        # Multiple videos
        videos = generator.create_continuous_stream(
            args.tracks,
            hours_per_video=args.hours,
            videos_to_generate=args.count
        )
        print(f"\n✅ {len(videos)} stream videos ready")
