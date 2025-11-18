"""
LoFi Music Empire - Main API Server

FastAPI server providing REST API and web UI for the complete
LoFi music generation and automation system.

Endpoints:
- /api/generate - Generate music tracks
- /api/videos - Manage video generation
- /api/schedule - Content scheduling
- /api/community - Community management
- /api/analytics - Analytics dashboard
- /api/copyright - Copyright checking
- /api/playlists - Playlist recommendations
- /api/status - System status

Web UI:
- / - Main dashboard
- /generate - Generation interface
- /content - Content management
- /analytics - Analytics view
- /settings - Configuration

Author: Claude
License: MIT
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import uuid
import numpy as np
import soundfile as sf
from dataclasses import asdict

# Import our modules
from src.metadata_generator import MetadataGenerator
from src.lofi_effects import LoFiEffectsChain
from src.ambient_sounds import AmbientSoundGenerator

# Initialize generation modules
metadata_gen = MetadataGenerator()
lofi_effects = LoFiEffectsChain()
ambient_gen = AmbientSoundGenerator()
SAMPLE_RATE = 44100

app = FastAPI(
    title="LoFi Music Empire API",
    description="Complete automation system for LoFi music creation and distribution",
    version="1.0.0"
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Data Models =====

class GenerateRequest(BaseModel):
    """Music generation request."""
    mood: str = "chill"
    duration: int = 180
    bpm: Optional[int] = None
    key: Optional[str] = None
    style: str = "lofi"
    count: int = 1


class VideoRequest(BaseModel):
    """Video generation request."""
    audio_path: str
    template: str = "classic_lofi"
    title: str
    artist: str = "LoFi AI"
    custom_background: Optional[str] = None


class ScheduleRequest(BaseModel):
    """Content scheduling request."""
    platform: str
    days_ahead: int = 30
    content_count: Optional[int] = None


class CommentRequest(BaseModel):
    """Process comment request."""
    comment_id: str
    platform: str
    author: str
    author_id: str
    text: str
    timestamp: str
    likes: int = 0
    replies: int = 0


class PlaylistRequest(BaseModel):
    """Playlist generation request."""
    mood: Optional[str] = None
    activity: Optional[str] = None
    duration_minutes: int = 60


class CopyrightCheckRequest(BaseModel):
    """Copyright check request."""
    melody_notes: List[int]
    melody_times: List[float]
    chords: List[str]
    key: str = "C"


# ===== State Management =====

class SystemState:
    """Global system state."""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.generated_tracks: List[Dict] = []
        self.scheduled_content: List[Dict] = []
        self.analytics: Dict = {
            'total_tracks': 0,
            'total_videos': 0,
            'total_uploads': 0,
            'total_comments_processed': 0
        }
        self.config: Dict = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration."""
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            'generation': {
                'default_mood': 'chill',
                'default_duration': 180,
                'default_bpm': 85
            },
            'video': {
                'default_template': 'classic_lofi',
                'width': 1920,
                'height': 1080,
                'fps': 60
            },
            'scheduling': {
                'youtube_posts_per_week': 3,
                'min_hours_between': 48
            },
            'community': {
                'auto_respond': True,
                'response_rate_positive': 0.3
            }
        }

    def create_job(self, job_type: str, params: Dict) -> str:
        """Create new job."""
        job_id = str(uuid.uuid4())[:8]
        self.jobs[job_id] = {
            'id': job_id,
            'type': job_type,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'params': params,
            'result': None,
            'error': None
        }
        return job_id

    def update_job(self, job_id: str, status: str = None,
                   progress: int = None, result: Any = None, error: str = None):
        """Update job status."""
        if job_id in self.jobs:
            if status:
                self.jobs[job_id]['status'] = status
            if progress is not None:
                self.jobs[job_id]['progress'] = progress
            if result is not None:
                self.jobs[job_id]['result'] = result
            if error is not None:
                self.jobs[job_id]['error'] = error
            self.jobs[job_id]['updated_at'] = datetime.now().isoformat()


state = SystemState()


# ===== API Endpoints =====

@app.get("/")
async def root():
    """Serve main dashboard."""
    return FileResponse("static/index.html")


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'analytics': state.analytics,
        'active_jobs': len([j for j in state.jobs.values() if j['status'] == 'running']),
        'pending_jobs': len([j for j in state.jobs.values() if j['status'] == 'pending']),
        'config': state.config
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return state.config


@app.post("/api/config")
async def update_config(config: Dict):
    """Update configuration."""
    state.config.update(config)
    # Save to file
    with open("config.json", 'w') as f:
        json.dump(state.config, f, indent=2)
    return {"status": "updated", "config": state.config}


@app.post("/api/generate")
async def generate_music(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate music tracks.

    Note: The actual generation model is proprietary.
    This endpoint integrates with your private generation backend.
    """
    job_id = state.create_job('generation', request.dict())

    # Add background task
    background_tasks.add_task(run_generation, job_id, request)

    return {
        'job_id': job_id,
        'status': 'pending',
        'message': f'Generation job created for {request.count} track(s)'
    }


def generate_single_track(mood: str, duration: int, bpm: Optional[int], key: Optional[str], output_path: Path) -> Dict:
    """Generate a single lo-fi track with real music composition."""
    # Setup
    sample_rate = SAMPLE_RATE
    audio = np.zeros(int(duration * sample_rate))
    t = np.arange(len(audio)) / sample_rate

    # Select key and BPM
    if not key:
        keys = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']
        key = np.random.choice(keys)
    if not bpm:
        bpm = 75

    # Key to frequency mapping
    key_freqs = {
        'C': 261.63, 'Cm': 261.63, 'D': 293.66, 'Dm': 293.66,
        'E': 329.63, 'Em': 329.63, 'F': 349.23, 'Fm': 349.23,
        'G': 392.00, 'Gm': 392.00, 'A': 440.00, 'Am': 440.00,
        'B': 493.88, 'Bm': 493.88
    }
    base_freq = key_freqs.get(key, 261.63)

    # Create chord progression
    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4
    num_bars = int(duration / bar_duration)

    chord_patterns = {
        'chill': [0, 5, 3, 4],
        'melancholic': [0, 3, 5, 7],
        'upbeat': [0, 4, 5, 7],
        'dreamy': [0, 5, 7, 3],
        'relaxed': [0, 4, 7, 5]
    }
    pattern = chord_patterns.get(mood, [0, 5, 3, 4])

    # Generate chords and bass
    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_duration
        bar_end = (bar_idx + 1) * bar_duration
        chord_degree = pattern[bar_idx % len(pattern)]

        chord_freq = base_freq * (2 ** (chord_degree / 12))

        bar_start_sample = int(bar_start * sample_rate)
        bar_end_sample = int(bar_end * sample_rate)
        if bar_end_sample > len(audio):
            bar_end_sample = len(audio)

        bar_t = t[bar_start_sample:bar_end_sample]
        envelope = np.exp(-2.5 * (bar_t - bar_start))

        # Chord (root, third, fifth)
        audio[bar_start_sample:bar_end_sample] += 0.08 * np.sin(2 * np.pi * chord_freq * bar_t) * envelope
        audio[bar_start_sample:bar_end_sample] += 0.06 * np.sin(2 * np.pi * chord_freq * 1.26 * bar_t) * envelope
        audio[bar_start_sample:bar_end_sample] += 0.05 * np.sin(2 * np.pi * chord_freq * 1.5 * bar_t) * envelope

        # Bass
        bass_freq = chord_freq / 2
        audio[bar_start_sample:bar_end_sample] += 0.15 * np.sin(2 * np.pi * bass_freq * bar_t) * envelope

        # Melody
        pentatonic = [0, 2, 4, 7, 9]
        for _ in range(np.random.randint(3, 7)):
            note_start = bar_start + np.random.uniform(0, bar_duration - beat_duration * 0.5)
            note_dur = beat_duration * np.random.choice([0.25, 0.5, 0.75])
            note_end = min(note_start + note_dur, bar_end)

            note_start_sample = int(note_start * sample_rate)
            note_end_sample = int(note_end * sample_rate)
            if note_end_sample > len(audio):
                break

            degree = np.random.choice(pentatonic)
            melody_freq = base_freq * (2 ** (degree / 12)) * (2 ** 2)
            note_t = t[note_start_sample:note_end_sample]
            note_envelope = np.exp(-4 * (note_t - note_start))
            audio[note_start_sample:note_end_sample] += 0.12 * np.sin(2 * np.pi * melody_freq * note_t) * note_envelope

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Apply lo-fi effects
    audio_lofi = lofi_effects.process_full_chain(audio, preset='medium')

    # Fade in/out
    fade_samples = int(0.5 * sample_rate)
    audio_lofi[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio_lofi[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    audio_lofi = audio_lofi / np.max(np.abs(audio_lofi)) * 0.75

    # Save audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_lofi, sample_rate)

    # Generate metadata
    metadata = metadata_gen.generate_complete_metadata(
        mood=mood,
        style='lofi',
        use_case='study',
        bpm=bpm,
        key=key,
        duration=duration
    )

    return {
        'audio_path': str(output_path),
        'metadata': asdict(metadata),
        'bpm': bpm,
        'key': key,
        'duration': duration
    }


async def run_generation(job_id: str, request: GenerateRequest):
    """Background task for music generation."""
    try:
        state.update_job(job_id, status='running', progress=0)

        generated_files = []
        output_dir = Path('output')

        for i in range(request.count):
            # Update progress
            progress = int((i + 1) / request.count * 100)
            state.update_job(job_id, progress=progress)

            # Generate real music track
            track_id = str(uuid.uuid4())[:8]
            audio_path = output_dir / 'audio' / f"{job_id}_{track_id}.wav"

            # Run generation in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            track_data = await loop.run_in_executor(
                None,
                generate_single_track,
                request.mood,
                request.duration,
                request.bpm,
                request.key,
                audio_path
            )

            track_info = {
                'track_id': track_id,
                'title': f"{request.mood.title()} Beats #{i+1}",
                'mood': request.mood,
                'duration': request.duration,
                'bpm': track_data['bpm'],
                'key': track_data['key'],
                'audio_path': track_data['audio_path'],
                'metadata': track_data['metadata'],
                'created_at': datetime.now().isoformat()
            }

            generated_files.append(track_info)
            state.generated_tracks.append(track_info)

        state.analytics['total_tracks'] += len(generated_files)
        state.update_job(job_id, status='completed', progress=100, result=generated_files)

    except Exception as e:
        import traceback
        print(f"Generation error: {str(e)}\n{traceback.format_exc()}")
        state.update_job(job_id, status='failed', error=str(e))


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.jobs[job_id]


@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List all jobs."""
    jobs = list(state.jobs.values())

    if status:
        jobs = [j for j in jobs if j['status'] == status]

    # Sort by created_at descending
    jobs.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        'total': len(jobs),
        'jobs': jobs[:limit]
    }


@app.post("/api/videos")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate video for track."""
    job_id = state.create_job('video', request.dict())
    background_tasks.add_task(run_video_generation, job_id, request)

    return {
        'job_id': job_id,
        'status': 'pending',
        'message': 'Video generation job created'
    }


async def run_video_generation(job_id: str, request: VideoRequest):
    """Background task for video generation."""
    try:
        state.update_job(job_id, status='running', progress=0)

        # Import video generator
        from src.video_generator import VideoGenerator, TemplateLibrary

        generator = VideoGenerator(
            width=state.config['video']['width'],
            height=state.config['video']['height'],
            fps=state.config['video']['fps']
        )

        template = TemplateLibrary.get_template(request.template)

        output_path = f"output/videos/{job_id}.mp4"

        # Generate video
        state.update_job(job_id, progress=50)

        success = generator.generate_video(
            audio_path=request.audio_path,
            output_path=output_path,
            template=template,
            title=request.title,
            artist=request.artist,
            custom_background=request.custom_background
        )

        if success:
            result = {
                'video_path': output_path,
                'title': request.title,
                'template': request.template
            }
            state.analytics['total_videos'] += 1
            state.update_job(job_id, status='completed', progress=100, result=result)
        else:
            state.update_job(job_id, status='failed', error='Video generation failed')

    except Exception as e:
        state.update_job(job_id, status='failed', error=str(e))


@app.post("/api/schedule")
async def create_schedule(request: ScheduleRequest):
    """Create content schedule."""
    try:
        from src.content_scheduler import ContentScheduler, Platform

        scheduler = ContentScheduler()

        # Convert platform string to enum
        platform = Platform[request.platform.upper()]

        calendar = scheduler.create_calendar(platform, request.days_ahead)

        # Get upcoming posts
        upcoming = calendar.get_next_posts(request.content_count or 10)

        schedule_data = [
            {
                'timestamp': slot.timestamp.isoformat(),
                'platform': slot.platform.value,
                'title': slot.title,
                'priority': slot.priority
            }
            for slot in upcoming
        ]

        return {
            'platform': request.platform,
            'days_ahead': request.days_ahead,
            'total_scheduled': len(schedule_data),
            'schedule': schedule_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/community/process")
async def process_comment(request: CommentRequest):
    """Process community comment."""
    try:
        from src.community_manager import CommunityManager, Comment, Platform

        manager = CommunityManager(dry_run=False)

        # Convert to Comment object
        comment = Comment(
            id=request.comment_id,
            platform=Platform[request.platform.upper()],
            author=request.author,
            author_id=request.author_id,
            text=request.text,
            timestamp=datetime.fromisoformat(request.timestamp),
            likes=request.likes,
            replies=request.replies
        )

        # Process comment
        manager.process_comment(comment)

        state.analytics['total_comments_processed'] += 1

        return {
            'status': 'processed',
            'comment_id': request.comment_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/community/insights")
async def get_community_insights():
    """Get community insights."""
    try:
        from src.community_manager import CommunityManager

        manager = CommunityManager()
        insights = manager.get_community_insights()

        return insights

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/playlists")
async def create_playlist(request: PlaylistRequest):
    """Create recommended playlist."""
    try:
        from src.playlist_recommender import PlaylistRecommender, Mood, Activity

        recommender = PlaylistRecommender()
        recommender.initialize_generator()

        # Convert strings to enums
        mood = Mood[request.mood.upper()] if request.mood else None
        activity = Activity[request.activity.upper()] if request.activity else None

        playlist = recommender.create_playlist(
            mood=mood,
            activity=activity,
            duration_minutes=request.duration_minutes
        )

        if playlist:
            return {
                'playlist_id': playlist.playlist_id,
                'name': playlist.name,
                'description': playlist.description,
                'track_count': len(playlist.tracks),
                'tracks': playlist.tracks
            }
        else:
            raise HTTPException(status_code=400, detail="Could not generate playlist")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/copyright/check")
async def check_copyright(request: CopyrightCheckRequest):
    """Check composition for copyright issues."""
    try:
        from src.copyright_protection import CopyrightDatabase, CopyrightProtector

        database = CopyrightDatabase()
        protector = CopyrightProtector(database)

        report = protector.check_composition(
            melody_notes=request.melody_notes,
            melody_times=request.melody_times,
            chords=request.chords,
            chord_key=request.key
        )

        return {
            'query_id': report.query_id,
            'risk_level': report.risk_level.value,
            'max_similarity': report.max_similarity,
            'is_safe': report.is_safe,
            'matches': [
                {
                    'work_id': m[0],
                    'similarity': m[1],
                    'component': m[2]
                }
                for m in report.matches
            ],
            'recommendations': report.recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics(days: int = 30):
    """Get analytics data."""
    try:
        # Return current analytics
        # In production, this would query the analytics database

        return {
            'period_days': days,
            'summary': state.analytics,
            'generated_tracks': state.generated_tracks[-10:],  # Last 10
            'scheduled_content': state.scheduled_content[-10:],  # Last 10
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracks")
async def list_tracks(limit: int = 50, mood: Optional[str] = None):
    """List generated tracks."""
    tracks = state.generated_tracks

    if mood:
        tracks = [t for t in tracks if t.get('mood') == mood]

    # Sort by created_at descending
    tracks.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    return {
        'total': len(tracks),
        'tracks': tracks[:limit]
    }


# ===== Serve Static Files (Web UI) =====

# Mount static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ===== Health Check =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    print("=" * 60)
    print("LoFi Music Empire API Server")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print(f"Config: {state.config}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API server...")


# ===== Run Server =====

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
