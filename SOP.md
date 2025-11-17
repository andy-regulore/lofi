# 🎵 LOFI MUSIC EMPIRE - STANDARD OPERATING PROCEDURES (SOP)

## Document Control

**Version:** 1.0
**Date:** 2025-11-17
**Status:** Production Ready
**Owner:** Music Empire Operations Team
**Review Cycle:** Quarterly

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Initial Setup](#2-initial-setup)
3. [Daily Operations](#3-daily-operations)
4. [Music Generation Workflow](#4-music-generation-workflow)
5. [Production & Mastering Workflow](#5-production--mastering-workflow)
6. [Content Creation Workflow](#6-content-creation-workflow)
7. [Distribution Workflow](#7-distribution-workflow)
8. [Analytics & Optimization](#8-analytics--optimization)
9. [Troubleshooting](#9-troubleshooting)
10. [Maintenance & Updates](#10-maintenance--updates)
11. [Quality Control](#11-quality-control)
12. [Emergency Procedures](#12-emergency-procedures)

---

## 1. System Overview

### 1.1 Purpose
This SOP defines standard procedures for operating the LoFi Music Empire AI system, covering music generation, production, distribution, and business analytics.

### 1.2 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LOFI MUSIC EMPIRE                        │
├─────────────────────────────────────────────────────────────┤
│  Generation → Production → Metadata → Distribution → Analytics│
└─────────────────────────────────────────────────────────────┘
```

**Core Modules:**
- Generation Engine (GPT-2, Diffusion, Style Transfer)
- Music Theory Engine (Jazz, Orchestration, Rhythm)
- Production Engine (Mixing, Mastering, LUFS)
- Business Automation (Metadata, Thumbnails, Uploads)
- Analytics Dashboard (Multi-platform tracking)

### 1.3 System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- 10GB storage
- CUDA-capable GPU (optional but recommended)

**Recommended:**
- Python 3.10+
- 32GB RAM
- 50GB SSD storage
- NVIDIA GPU with 8GB+ VRAM
- Docker & Docker Compose

---

## 2. Initial Setup

### 2.1 Environment Setup

**Step 1: Clone Repository**
```bash
git clone https://github.com/andy-regulore/lofi.git
cd lofi
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-prod.txt  # For production features
pip install -e .  # Install as editable package
```

**Step 4: Configure Environment Variables**
```bash
cp .env.example .env
nano .env  # Edit with your settings
```

Required environment variables:
```bash
# Model Configuration
MODEL_PATH=./models/lofi_model.pt
TOKENIZER_PATH=./models/tokenizer.json

# API Keys (optional for automation)
YOUTUBE_API_KEY=your_youtube_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Output Paths
OUTPUT_DIR=./output
TEMP_DIR=./temp

# Production Settings
TARGET_LUFS=-14.0
SAMPLE_RATE=44100
```

**Step 5: Download Pre-trained Models**
```bash
python scripts/download_models.py
```

**Step 6: Verify Installation**
```bash
python -m pytest tests/
```

Expected output: All tests passing (80%+ coverage)

### 2.2 Docker Setup (Recommended)

**Step 1: Build Docker Images**
```bash
docker-compose build
```

**Step 2: Start Services**
```bash
docker-compose up -d
```

**Step 3: Verify Services**
```bash
docker-compose ps
```

Expected services:
- `dev` (Development environment)
- `api` (REST API server)
- `jupyter` (Notebook server)
- `tensorboard` (Training monitoring)
- `prometheus` (Metrics)
- `grafana` (Dashboards)

### 2.3 Initial Configuration

**Step 1: Configure Music Generation Settings**

Edit `config.yaml`:
```yaml
generation:
  model_type: "gpt2"
  model_size: "medium"  # 117M parameters
  max_length: 2048
  temperature: 0.9
  top_p: 0.95

conditioning:
  default_tempo: 75
  default_key: "Am"
  default_mood: "chill"

quality:
  min_score: 0.7
  auto_reject: true
```

**Step 2: Configure Production Settings**

Edit `config.yaml`:
```yaml
production:
  mastering:
    target_lufs: -14.0  # Streaming standard
    preset: "lofi"
    ceiling: -0.3

  mixing:
    apply_eq: true
    apply_compression: true
    stereo_width: 0.95
```

**Step 3: Configure Business Automation**

Edit `config.yaml`:
```yaml
business:
  metadata:
    language: "en"
    seo_optimization: true
    include_timestamps: true

  youtube:
    default_privacy: "public"
    auto_playlist: true
    upload_time: "18:00"  # 6 PM

  analytics:
    track_youtube: true
    track_spotify: true
    update_interval: 3600  # 1 hour
```

---

## 3. Daily Operations

### 3.1 Daily Startup Checklist

**Time Required:** 10 minutes

**☐ Step 1: System Health Check**
```bash
# Check all services are running
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check disk space (need 10GB+ free)
df -h
```

**☐ Step 2: Review Overnight Generation**
```bash
# Check generation queue status
python scripts/check_queue.py

# Review generated tracks
python scripts/review_tracks.py --date today
```

**☐ Step 3: Analytics Review**
```bash
# Generate daily report
python scripts/daily_report.py

# Check for quality drift
python scripts/check_drift.py
```

**☐ Step 4: Upload Status Check**
```bash
# Check pending uploads
python scripts/check_uploads.py

# Verify latest uploads
python scripts/verify_uploads.py --hours 24
```

### 3.2 Daily Task Schedule

**Morning (9:00 AM)**
- Review analytics dashboard
- Check overnight generation results
- Approve/reject generated tracks
- Plan content for the day

**Midday (12:00 PM)**
- Generate new batch of tracks (10-20)
- Create thumbnails for approved tracks
- Generate metadata
- Queue uploads for evening

**Afternoon (3:00 PM)**
- Review A/B test results
- Optimize upload schedule
- Update playlists
- Respond to top comments

**Evening (6:00 PM - Prime Time)**
- Automated uploads go live
- Monitor initial performance
- Engage with audience

**Night (10:00 PM)**
- Queue overnight batch generation (50-100 tracks)
- Set up A/B tests for tomorrow
- Review day's performance

---

## 4. Music Generation Workflow

### 4.1 Single Track Generation

**Time Required:** 2-5 minutes per track

**Step 1: Define Parameters**
```python
from src.generator import LoFiGenerator
from src.model import ConditionedLoFiModel
from src.tokenizer import LoFiTokenizer
import yaml

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize
tokenizer = LoFiTokenizer(config)
model = ConditionedLoFiModel(config, tokenizer.get_vocab_size())
generator = LoFiGenerator(model, tokenizer, config)
```

**Step 2: Generate Track**
```python
# Generate with specific parameters
tokens, metadata = generator.generate_track(
    tempo=75,           # BPM (60-95 for lofi)
    key='Am',          # Musical key
    mood='peaceful',   # Mood/vibe
    length=1024,       # Token length (affects duration)
    temperature=0.9    # Creativity (0.7-1.0)
)

# Save as MIDI
output_path = generator.tokens_to_midi(tokens, 'output/track_001.mid')
print(f"Generated: {output_path}")
```

**Step 3: Quality Check**
```python
from src.generator import QualityScorer

scorer = QualityScorer()
score = scorer.score_track(tokens, metadata)

print(f"Quality Score: {score:.2f}")
print(f"Melodic Coherence: {metadata['melodic_coherence']:.2f}")
print(f"Harmonic Variety: {metadata['harmonic_variety']:.2f}")

# Auto-reject if below threshold
if score < 0.7:
    print("REJECTED: Below quality threshold")
else:
    print("APPROVED: Ready for production")
```

### 4.2 Batch Generation

**Time Required:** 1-2 hours for 50 tracks (overnight)

**Step 1: Create Generation Queue**
```python
from src.cli import LoFiCLI

cli = LoFiCLI()

# Define batch parameters
variations = [
    {'tempo': 70, 'key': 'Am', 'mood': 'melancholic'},
    {'tempo': 75, 'key': 'Dm', 'mood': 'peaceful'},
    {'tempo': 80, 'key': 'C', 'mood': 'uplifting'},
    {'tempo': 72, 'key': 'Em', 'mood': 'dreamy'},
    {'tempo': 78, 'key': 'G', 'mood': 'cozy'},
]

# Generate multiple tracks with variations
for i, params in enumerate(variations):
    for j in range(10):  # 10 tracks per variation = 50 total
        cli.generate(
            output=f'output/batch_{i}_{j}.mid',
            **params
        )
```

**Step 2: Automated Quality Filtering**
```bash
# Run batch quality check
python scripts/batch_quality_check.py \
    --input output/ \
    --threshold 0.7 \
    --move-rejects rejected/
```

**Step 3: Review Results**
```bash
# Generate batch report
python scripts/batch_report.py --input output/

# Output:
# Total Generated: 50
# Approved: 38 (76%)
# Rejected: 12 (24%)
# Average Score: 0.82
```

### 4.3 Advanced Generation Techniques

#### 4.3.1 Style Transfer
```python
from src.style_transfer import StyleTransferModel

# Load reference track
reference_audio = load_audio('references/jazzy_lofi.wav')

# Transfer style to your track
model = StyleTransferModel()
styled_tokens = model(content_tokens, style_music=reference_audio)

# Save result
generator.tokens_to_midi(styled_tokens, 'output/styled_track.mid')
```

#### 4.3.2 Diffusion-Based Generation
```python
from src.diffusion_models import DiffusionModel, UNet1D, DiffusionConfig

# Configure diffusion
config = DiffusionConfig(
    num_timesteps=1000,
    noise_schedule='cosine'
)

unet = UNet1D(in_channels=128, model_channels=256)
diffusion = DiffusionModel(unet, config)

# Generate
samples = diffusion.sample(
    shape=(1, 128, 256),
    num_steps=50  # Faster sampling
)
```

#### 4.3.3 RLHF Fine-tuning
```python
from src.rlhf import RewardModelTrainer, PPOTrainer

# Collect human preferences
# Format: [{"track_a": [...], "track_b": [...], "preferred": "a"}]
preferences = load_preferences('data/preferences.json')

# Train reward model
reward_trainer = RewardModelTrainer(model)
reward_trainer.train(preferences, epochs=10)

# Fine-tune with PPO
ppo_trainer = PPOTrainer(model, reward_trainer.reward_model)
ppo_trainer.train(num_episodes=100)
```

---

## 5. Production & Mastering Workflow

### 5.1 Convert MIDI to Audio

**Step 1: Render MIDI with Virtual Instruments**
```bash
# Using FluidSynth (free, open-source)
fluidsynth -ni soundfont.sf2 track.mid -F track_raw.wav -r 44100

# Or using professional tools:
# - Ableton Live (with export automation)
# - Logic Pro (with batch bounce)
# - Kontakt (with scripting)
```

**Step 2: Load and Verify Audio**
```python
import soundfile as sf
import numpy as np

# Load audio
audio, sample_rate = sf.read('track_raw.wav')
print(f"Duration: {len(audio)/sample_rate:.2f}s")
print(f"Sample Rate: {sample_rate}Hz")
print(f"Channels: {audio.shape}")
```

### 5.2 Professional Mixing

**Step 1: Apply EQ**
```python
from src.mixing_mastering import ParametricEQ, EQBand

eq = ParametricEQ(num_bands=7)

# Manual EQ
eq.bands[0] = EQBand(frequency=80, gain=-3.0, q_factor=0.7, filter_type='shelf')
eq.bands[1] = EQBand(frequency=200, gain=2.0, q_factor=1.2, filter_type='peak')
eq.bands[2] = EQBand(frequency=3000, gain=-1.5, q_factor=1.5, filter_type='peak')
eq.bands[3] = EQBand(frequency=8000, gain=-2.0, q_factor=0.8, filter_type='shelf')

audio_eq = eq.process(audio, sample_rate)

# Or use auto-EQ
suggestions = eq.auto_eq(audio, sample_rate, target_curve='warm')
for suggestion in suggestions:
    eq.add_band(suggestion)
audio_eq = eq.process(audio, sample_rate)
```

**Step 2: Apply Compression**
```python
from src.mixing_mastering import MultiBandCompressor

compressor = MultiBandCompressor(num_bands=4)
audio_compressed = compressor.process(audio_eq, sample_rate)
```

**Step 3: Adjust Stereo Width** (if stereo)
```python
from src.mixing_mastering import StereoImaging

# Analyze current stereo field
analysis = StereoImaging.analyze_stereo_field(audio_compressed)
print(f"Width: {analysis['width']:.3f}")
print(f"Correlation: {analysis['correlation']:.3f}")

# Adjust width (0.95 for lofi - slightly narrower)
audio_stereo = StereoImaging.adjust_width(audio_compressed, width=0.95)
```

### 5.3 Professional Mastering

**Full Automated Mastering Chain:**
```python
from src.mixing_mastering import MasteringChain

# Initialize mastering chain
chain = MasteringChain()

# Master for streaming platforms (Spotify, Apple Music, YouTube)
mastered = chain.master(
    audio_stereo,
    sample_rate=44100,
    preset='streaming'  # or 'cd', 'vinyl', 'club'
)

# Analyze final master
analysis = chain.analyze_master(mastered, sample_rate)

print("=== Master Analysis ===")
print(f"Integrated LUFS: {analysis['lufs_integrated']:.1f} (Target: -14.0)")
print(f"True Peak: {analysis['true_peak_db']:.1f} dBFS")
print(f"Dynamic Range: {analysis['dynamic_range_db']:.1f} dB")
print(f"Stereo Width: {analysis.get('width', 'N/A'):.3f}")
print(f"Correlation: {analysis.get('correlation', 'N/A'):.3f}")
```

**Quality Assurance Checks:**
```python
# Check if meets standards
def verify_master_quality(analysis):
    checks = []

    # LUFS tolerance
    target_lufs = -14.0
    lufs_diff = abs(analysis['lufs_integrated'] - target_lufs)
    checks.append(('LUFS Target', lufs_diff < 0.5, f"{lufs_diff:.1f}dB"))

    # True peak headroom
    checks.append(('True Peak', analysis['true_peak_db'] < -0.1, f"{analysis['true_peak_db']:.1f}dBFS"))

    # Dynamic range
    checks.append(('Dynamic Range', analysis['dynamic_range_db'] > 6.0, f"{analysis['dynamic_range_db']:.1f}dB"))

    # Stereo correlation (if stereo)
    if 'correlation' in analysis:
        checks.append(('Stereo Correlation', 0.5 < analysis['correlation'] < 1.0, f"{analysis['correlation']:.3f}"))

    # Print results
    print("\n=== Quality Checks ===")
    all_passed = True
    for name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}: {value}")
        all_passed = all_passed and passed

    return all_passed

passed = verify_master_quality(analysis)
if passed:
    print("\n✅ Master approved for distribution")
else:
    print("\n❌ Master needs adjustment")
```

**Step 4: Export Final Master**
```python
import soundfile as sf

# Export high-quality master
sf.write(
    'output/track_001_mastered.wav',
    mastered,
    sample_rate,
    subtype='PCM_24'  # 24-bit for archival
)

# Export streaming version (16-bit)
sf.write(
    'output/track_001_streaming.wav',
    mastered,
    sample_rate,
    subtype='PCM_16'
)

# Convert to MP3 for uploads
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'output/track_001_streaming.wav',
    '-codec:a', 'libmp3lame', '-qscale:a', '2',  # VBR ~190kbps
    'output/track_001.mp3'
])
```

### 5.4 Batch Production

**Automated Batch Processing:**
```bash
# Process entire batch
python scripts/batch_production.py \
    --input output/*.mid \
    --output mastered/ \
    --preset streaming \
    --verify-quality \
    --parallel 4  # Process 4 tracks in parallel
```

**Custom Batch Script:**
```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from src.mixing_mastering import MasteringChain
import soundfile as sf

def process_track(midi_path):
    """Process single track through full production pipeline."""

    # 1. Render MIDI to audio
    audio_path = render_midi(midi_path)  # Your rendering function

    # 2. Load audio
    audio, sr = sf.read(audio_path)

    # 3. Master
    chain = MasteringChain()
    mastered = chain.master(audio, sr, preset='streaming')

    # 4. Quality check
    analysis = chain.analyze_master(mastered, sr)
    if verify_master_quality(analysis):
        # 5. Export
        output_path = f"mastered/{midi_path.stem}.wav"
        sf.write(output_path, mastered, sr, subtype='PCM_16')
        return output_path
    else:
        return None

# Process batch in parallel
midi_files = list(Path('output').glob('*.mid'))
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_track, midi_files))

# Report
successful = [r for r in results if r is not None]
print(f"Successfully mastered: {len(successful)}/{len(midi_files)}")
```

---

## 6. Content Creation Workflow

### 6.1 Metadata Generation

**Step 1: Generate Complete Metadata**
```python
from src.metadata_generator import MetadataGenerator

generator = MetadataGenerator()

# Generate metadata for track
metadata = generator.generate_complete_metadata(
    mood='chill',
    style='lofi',
    use_case='study',
    bpm=75,
    key='Am',
    duration=3600,  # 60 minutes
    seasonal='winter'  # Optional seasonal tags
)

print("=== Generated Metadata ===")
print(f"Title: {metadata.title}")
print(f"Tags: {', '.join(metadata.tags[:10])}")
print(f"\nDescription:\n{metadata.description[:200]}...")
```

**Step 2: A/B Test Variations**
```python
# Generate 3 title variations for testing
variations = generator.generate_ab_test_variations(
    mood='chill',
    style='lofi',
    use_case='study',
    num_variations=3
)

print("=== A/B Test Variations ===")
for i, (title, desc_style) in enumerate(variations, 1):
    print(f"\nVariation {i}:")
    print(f"  Title: {title}")
    print(f"  Style: {desc_style}")
```

**Step 3: Organize into Playlists**
```python
from src.metadata_generator import PlaylistOrganizer

organizer = PlaylistOrganizer()

# Create mood-based playlists
mood_playlists = organizer.organize_by_mood(tracks_metadata)

# Create seasonal playlists
seasonal_playlists = organizer.organize_by_season(tracks_metadata)

# Create series (e.g., "30 Days of Study Beats")
series = organizer.create_series_playlists(
    "30 Days of Study Beats",
    tracks_per_playlist=3,
    all_tracks=all_track_titles
)
```

### 6.2 Thumbnail Generation

**Step 1: Generate Single Thumbnail**
```python
from src.youtube_thumbnail import (
    ThumbnailGenerator,
    ThumbnailConfig,
    ThumbnailStyle,
    ColorGrading
)

generator = ThumbnailGenerator()

config = ThumbnailConfig(
    width=1280,
    height=720,
    title_text="Chill Lofi Beats",
    subtitle_text="Study & Relax",
    style=ThumbnailStyle.ANIME,
    color_grading=ColorGrading.WARM,
    overlay_opacity=0.4,
    add_logo=True,
    add_border=False
)

thumbnail = generator.generate_thumbnail(config)
thumbnail.save('thumbnails/track_001.png')
```

**Step 2: Batch Generate Thumbnails**
```python
# Generate thumbnails for multiple tracks
titles = [
    "Morning Coffee Vibes - Chill Lofi",
    "Late Night Study Session",
    "Rainy Day Ambience",
    "Peaceful Morning Beats"
]

thumbnails = generator.batch_generate(
    titles,
    style=ThumbnailStyle.MINIMAL,
    color=ColorGrading.COOL
)

for i, (title, thumb) in enumerate(zip(titles, thumbnails)):
    thumb.save(f'thumbnails/track_{i:03d}.png')
```

**Step 3: A/B Test Thumbnails**
```python
# Generate 3 variations for testing
variations = generator.generate_ab_test_variations(
    title="Study Session",
    num_variations=3
)

for i, thumbnail in enumerate(variations, 1):
    thumbnail.save(f'thumbnails/test_variation_{i}.png')
```

### 6.3 Video Creation

**Step 1: Create Static Video from Audio**
```bash
# Create video with still image and audio
ffmpeg -loop 1 -i thumbnail.png -i audio.mp3 \
    -c:v libx264 -tune stillimage -c:a aac -b:a 192k \
    -pix_fmt yuv420p -shortest -t 3600 \
    output_video.mp4
```

**Step 2: Create Animated Visualizer** (Advanced)
```bash
# Create audio visualizer
ffmpeg -i audio.mp3 -filter_complex \
    "[0:a]showwaves=s=1280x720:mode=line:rate=25,colorkey=0x000000:0.01:0.1,format=yuva420p[vid]" \
    -map "[vid]" -map 0:a -c:v libx264 -c:a copy \
    output_visualizer.mp4
```

**Step 3: Add Background Loop** (Optional)
```bash
# Loop background video with audio
ffmpeg -stream_loop -1 -i background_loop.mp4 -i audio.mp3 \
    -c:v copy -c:a aac -b:a 192k -shortest \
    -t 3600 output_video.mp4
```

---

## 7. Distribution Workflow

### 7.1 YouTube Upload

**Step 1: Prepare Upload**
```python
from src.youtube_automation import (
    YouTubeUploader,
    VideoMetadata,
    UploadSchedule
)
from datetime import datetime, timedelta

# Initialize uploader
uploader = YouTubeUploader(
    api_key='your_youtube_api_key',
    oauth_credentials={'credentials': 'path/to/oauth.json'}
)

# Create metadata
metadata = VideoMetadata(
    title="Chill Lofi Beats to Study/Relax To [Peaceful Vibes]",
    description=generated_description,
    tags=['lofi', 'study music', 'chill beats', 'focus music', 'relaxing'],
    category_id="10",  # Music
    privacy_status="public",
    thumbnail_path="thumbnails/track_001.png",
    playlist_ids=['playlist_id_1', 'playlist_id_2']
)
```

**Step 2: Schedule Upload**
```python
# Schedule for optimal time (e.g., 6 PM today)
schedule = UploadSchedule(
    video_path="videos/track_001.mp4",
    metadata=metadata,
    scheduled_time=datetime.now().replace(hour=18, minute=0),
    priority=5
)

uploader.add_to_queue(schedule)
```

**Step 3: Process Upload Queue**
```python
# Process all due uploads
results = uploader.process_upload_queue(max_uploads=10)

for result in results:
    print(f"✅ Uploaded: {result['title']}")
    print(f"   URL: {result['url']}")
    print(f"   Video ID: {result['video_id']}")
```

**Step 4: Verify Upload**
```bash
# Check upload status
python scripts/verify_uploads.py --hours 1

# Monitor first hour performance
python scripts/monitor_video.py --video-id VIDEO_ID --duration 3600
```

### 7.2 Playlist Management

**Step 1: Create Mood Playlists**
```python
from src.youtube_automation import PlaylistManager

manager = PlaylistManager(api_key='your_key')

# Create seasonal playlists
seasonal = manager.organize_by_season()

# Create mood playlists
mood_videos = [
    {'title': 'Track 1', 'video_id': 'vid1', 'mood': 'peaceful'},
    {'title': 'Track 2', 'video_id': 'vid2', 'mood': 'peaceful'},
    {'title': 'Track 3', 'video_id': 'vid3', 'mood': 'uplifting'},
]

mood_playlists = manager.organize_by_mood(mood_videos)
```

**Step 2: Create Series**
```python
# Create "30 Days of Study Beats" series
series = manager.create_series(
    base_title="30 Days of Study Beats",
    num_episodes=30
)

print(f"Created {len(series)} playlists")
```

**Step 3: Update Playlists**
```python
# Add videos to playlists
for playlist_id in ['pl_1', 'pl_2']:
    manager.update_playlist(playlist_id, video_ids=['vid1', 'vid2', 'vid3'])
```

### 7.3 Multi-Platform Distribution

**Spotify/Apple Music via DistroKid:**
```python
# Note: Requires DistroKid API access (paid tier)

from src.distribution import DistributionManager

distributor = DistributionManager(
    distrokid_api_key='your_key'
)

# Submit release
release = distributor.submit_release(
    title="Peaceful Lofi Collection",
    artist="Your Artist Name",
    tracks=[
        {'file': 'track_001.wav', 'title': 'Morning Vibes'},
        {'file': 'track_002.wav', 'title': 'Afternoon Chill'},
    ],
    album_art='album_cover.jpg',
    release_date='2025-12-01',
    platforms=['spotify', 'apple_music', 'youtube_music']
)

print(f"Release submitted: {release['release_id']}")
```

### 7.4 Optimal Upload Timing

**Analyze Best Upload Times:**
```python
# Get optimal times based on audience analytics
optimal_times = uploader.get_optimal_upload_times(timezone='America/New_York')

print("=== Optimal Upload Times ===")
for day, hour in optimal_times:
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"{days[day]} at {hour}:00")

# Output example:
# Monday at 18:00
# Wednesday at 18:00
# Friday at 18:00
# Saturday at 10:00
# Sunday at 10:00
```

**Schedule Week of Uploads:**
```python
from datetime import datetime, timedelta

tracks_to_upload = [...]  # List of prepared tracks
optimal_times = [...] # From above

# Schedule uploads throughout week
for i, (track, (day, hour)) in enumerate(zip(tracks_to_upload, optimal_times)):
    # Calculate next occurrence of day/hour
    now = datetime.now()
    days_ahead = day - now.weekday()
    if days_ahead < 0:
        days_ahead += 7

    upload_time = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=0)

    schedule = UploadSchedule(
        video_path=track['video'],
        metadata=track['metadata'],
        scheduled_time=upload_time,
        priority=i
    )

    uploader.add_to_queue(schedule)
    print(f"Scheduled: {track['title']} for {upload_time}")
```

---

## 8. Analytics & Optimization

### 8.1 Daily Analytics Review

**Step 1: Generate Daily Report**
```python
from src.analytics_dashboard import MasterDashboard

dashboard = MasterDashboard()

# Get overview
overview = dashboard.get_overview()

print(dashboard.generate_report())
```

Expected output:
```
============================================================
LOFI MUSIC EMPIRE - ANALYTICS DASHBOARD
============================================================

YOUTUBE PERFORMANCE:
  Videos: 150
  Total Views: 1,234,567
  Subscribers: 45,678
  Average CPM: $2.45

SPOTIFY PERFORMANCE:
  Tracks: 50
  Total Streams: 567,890
  Monthly Listeners: 12,345

FINANCIAL OVERVIEW (Last 30 Days):
  Revenue: $3,456.78
  Costs: $234.56
  Profit: $3,222.22
  ROI: 1374.2%

GROWTH METRICS:
  Subscriber Growth (30d): 15.3%
  100K Subscribers ETA: 2026-04-15

============================================================
```

**Step 2: Track Top Performing Content**
```python
# Get top videos
top_videos = dashboard.youtube.get_top_videos(n=10, metric='views')

print("=== Top 10 Videos by Views ===")
for i, video in enumerate(top_videos, 1):
    print(f"{i}. {video.title}")
    print(f"   Views: {video.views_or_streams:,}")
    print(f"   Revenue: ${video.revenue_usd:.2f}")
    print()
```

**Step 3: Analyze Upload Patterns**
```python
# Find best performing upload times
patterns = dashboard.youtube.analyze_upload_patterns()

print("=== Upload Pattern Analysis ===")
print(f"Best Day: {patterns['best_day']}")
print(f"Best Hour: {patterns['best_hour']}:00")
print("\nPerformance by Day:")
for day, stats in patterns['by_day'].items():
    print(f"  {day}: {stats['avg_views']:,.0f} avg views")
```

### 8.2 A/B Testing

**Step 1: Set Up A/B Test**
```python
from src.music_analysis import ABTestingFramework

ab_test = ABTestingFramework()

# Record comparisons (user feedback or performance data)
ab_test.record_comparison(
    variant_a_id='title_variation_1',
    variant_b_id='title_variation_2',
    winner='a',  # or 'b' or 'tie'
    metrics={'views': 1000, 'ctr': 0.05}
)

# ... collect more data points
```

**Step 2: Analyze Results**
```python
# Get win rates
win_rates = ab_test.get_win_rates()

print("=== A/B Test Results ===")
for variant, rates in win_rates.items():
    print(f"{variant}:")
    print(f"  Wins: {rates['wins']}")
    print(f"  Losses: {rates['losses']}")
    print(f"  Win Rate: {rates['win_rate']:.1%}")

# Check statistical significance
p_value, significant, winner = ab_test.statistical_significance(
    'title_variation_1',
    'title_variation_2',
    alpha=0.05
)

if significant:
    print(f"\n✅ Results are statistically significant (p={p_value:.4f})")
    print(f"Winner: {winner}")
else:
    print(f"\n⚠️ Not enough data for significance (p={p_value:.4f})")
```

### 8.3 Revenue Tracking

**Step 1: Record Revenue**
```python
from datetime import datetime

# Add revenue entries
dashboard.financial.add_revenue('youtube_ads', 123.45, datetime.now())
dashboard.financial.add_revenue('spotify_streams', 67.89, datetime.now())
dashboard.financial.add_revenue('sample_packs', 200.00, datetime.now())

# Add costs
dashboard.financial.add_cost('distribution', 19.99, datetime.now(), "DistroKid monthly")
dashboard.financial.add_cost('api_costs', 5.00, datetime.now(), "YouTube API")
```

**Step 2: Calculate Metrics**
```python
from datetime import timedelta

# Last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

revenue = dashboard.financial.get_total_revenue(start_date, end_date)
costs = dashboard.financial.get_total_costs(start_date, end_date)
roi = dashboard.financial.calculate_roi(start_date, end_date)

print(f"30-Day Revenue: ${revenue:.2f}")
print(f"30-Day Costs: ${costs:.2f}")
print(f"30-Day Profit: ${revenue - costs:.2f}")
print(f"ROI: {roi:.1f}%")
```

**Step 3: Project Future Revenue**
```python
# Project next 6 months
projections = dashboard.financial.project_revenue(months=6)

print("\n=== Revenue Projections ===")
for month, projected in projections:
    print(f"{month}: ${projected:.2f}")
```

### 8.4 Growth Tracking

**Step 1: Add Growth Data**
```python
# Track metrics over time
dashboard.growth.add_data_point('subscribers', datetime.now(), 45678)
dashboard.growth.add_data_point('monthly_listeners', datetime.now(), 12345)
dashboard.growth.add_data_point('total_views', datetime.now(), 1234567)
```

**Step 2: Calculate Growth Rates**
```python
# 30-day growth rate
subscriber_growth = dashboard.growth.calculate_growth_rate('subscribers', days=30)
listener_growth = dashboard.growth.calculate_growth_rate('monthly_listeners', days=30)

print(f"Subscriber Growth (30d): {subscriber_growth:.1f}%")
print(f"Listener Growth (30d): {listener_growth:.1f}%")
```

**Step 3: Project Milestones**
```python
# When will we hit 100K subscribers?
eta_100k = dashboard.growth.project_milestone('subscribers', 100000)

if eta_100k:
    print(f"100K Subscribers ETA: {eta_100k.strftime('%Y-%m-%d')}")
    days_to_go = (eta_100k - datetime.now()).days
    print(f"Days to go: {days_to_go}")
```

### 8.5 Quality Monitoring

**Step 1: Monitor Generation Quality**
```python
from src.music_analysis import QualityDashboard

quality_dashboard = QualityDashboard()

# Record each generation
quality_dashboard.record_generation(
    generation_id='gen_001',
    metrics={
        'quality_score': 0.85,
        'melodic_coherence': 0.90,
        'harmonic_variety': 0.80
    },
    metadata={'tempo': 75, 'key': 'Am', 'mood': 'peaceful'}
)
```

**Step 2: Detect Quality Drift**
```python
# Check if quality is degrading over time
drift = quality_dashboard.detect_quality_drift(
    metric_name='quality_score',
    window_size=100,
    threshold=0.1
)

if drift['drift_detected']:
    print(f"⚠️ Quality drift detected!")
    print(f"Relative change: {drift['relative_change']:.1%}")
    print(f"Action: Review model, retrain if necessary")
else:
    print("✅ Quality stable")
```

---

## 9. Troubleshooting

### 9.1 Common Issues

#### Issue 1: Low Quality Scores

**Symptoms:**
- Generated tracks scoring below 0.7
- High rejection rate in batch generation

**Diagnosis:**
```python
# Analyze failed tracks
from src.generator import QualityScorer

scorer = QualityScorer()
for track_path in failed_tracks:
    tokens = load_tokens(track_path)
    score, details = scorer.score_track_detailed(tokens)

    print(f"Track: {track_path}")
    print(f"  Overall: {score:.2f}")
    print(f"  Melodic Coherence: {details['melodic_coherence']:.2f}")
    print(f"  Harmonic Variety: {details['harmonic_variety']:.2f}")
    print(f"  Rhythmic Stability: {details['rhythmic_stability']:.2f}")
```

**Solutions:**
1. **Adjust temperature:** Lower for more conservative output
   ```python
   tokens = generator.generate_track(..., temperature=0.8)  # Was 0.9
   ```

2. **Use constrained decoding:**
   ```python
   from src.optimization import ConstrainedDecoder

   decoder = ConstrainedDecoder(tokenizer)
   decoder.add_music_theory_constraints()
   tokens = decoder.decode(model_output)
   ```

3. **Retrain with curriculum learning:**
   ```python
   from src.curriculum_learning import CurriculumTrainer

   trainer = CurriculumTrainer(model)
   trainer.train(dataset, start_difficulty=0.3, end_difficulty=0.8)
   ```

#### Issue 2: LUFS Not Meeting Target

**Symptoms:**
- Mastered tracks too loud or too quiet
- LUFS outside ±1dB of target

**Diagnosis:**
```python
from src.mixing_mastering import LoudnessProcessor

# Measure actual LUFS
loudness = LoudnessProcessor.measure_lufs(audio, sample_rate)
print(f"Current LUFS: {loudness['lufs_integrated']:.1f}")
print(f"Target LUFS: -14.0")
print(f"Difference: {loudness['lufs_integrated'] - (-14.0):.1f}dB")
```

**Solutions:**
1. **Re-run normalization:**
   ```python
   audio_normalized = LoudnessProcessor.normalize_lufs(
       audio,
       target_lufs=-14.0,
       sample_rate=44100
   )
   ```

2. **Adjust mastering preset:**
   ```python
   chain = MasteringChain()
   chain.target_lufs = -14.0  # Ensure correct target
   chain.limiter_settings.ceiling = -0.3  # More headroom
   ```

3. **Check for clipping in source:**
   ```python
   peak = np.max(np.abs(audio))
   if peak > 0.99:
       print("⚠️ Source audio is clipping!")
       # Reduce gain before mastering
       audio = audio * 0.8
   ```

#### Issue 3: Upload Failures

**Symptoms:**
- Videos failing to upload
- API errors

**Diagnosis:**
```bash
# Check API quota
python scripts/check_api_quota.py

# Check video file
ffprobe video.mp4

# Test API connection
python scripts/test_youtube_api.py
```

**Solutions:**
1. **Quota exceeded:** Wait for quota reset or upgrade account
2. **File format issues:**
   ```bash
   # Re-encode video
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac \
       -strict -2 -preset medium output.mp4
   ```

3. **OAuth token expired:**
   ```python
   # Refresh OAuth credentials
   python scripts/refresh_oauth.py
   ```

#### Issue 4: Memory Issues

**Symptoms:**
- Out of memory errors
- System slowdown during batch processing

**Solutions:**
1. **Reduce batch size:**
   ```python
   # Process smaller batches
   batch_size = 10  # Was 50
   ```

2. **Use model quantization:**
   ```python
   from src.optimization import ModelQuantizer

   quantizer = ModelQuantizer()
   quantized_model = quantizer.quantize(model, precision='int8')
   ```

3. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

4. **Process sequentially instead of parallel:**
   ```python
   # Instead of ProcessPoolExecutor
   for track in tracks:
       process_track(track)
   ```

### 9.2 Performance Optimization

**Issue: Slow Generation**

**Solutions:**
1. **Use KV-cache:**
   ```python
   generator.use_kv_cache = True
   ```

2. **Reduce max length:**
   ```python
   tokens = generator.generate_track(..., length=512)  # Was 1024
   ```

3. **Use beam search width:**
   ```python
   tokens = generator.generate_track(..., num_beams=3)  # Was 5
   ```

**Issue: Slow Mastering**

**Solutions:**
1. **Process at lower sample rate:**
   ```python
   # Downsample for processing
   audio_44k = librosa.resample(audio, orig_sr=48000, target_sr=44100)

   # Master
   mastered = chain.master(audio_44k, sample_rate=44100)

   # Upsample back if needed
   ```

2. **Use simpler mastering preset:**
   ```python
   # Use 'streaming' instead of 'cd' (fewer processing stages)
   mastered = chain.master(audio, preset='streaming')
   ```

### 9.3 Support Contacts

**For Technical Issues:**
- GitHub Issues: https://github.com/andy-regulore/lofi/issues
- Documentation: See README.md, GUIDE.md, USAGE.md

**For API Issues:**
- YouTube API: https://developers.google.com/youtube/v3/support
- Spotify API: https://developer.spotify.com/support

---

## 10. Maintenance & Updates

### 10.1 Daily Maintenance

**Morning Checklist (5 minutes):**
```bash
☐ Check system health
☐ Review overnight generations
☐ Clear temp files: rm -rf temp/*
☐ Check disk space: df -h
☐ Review error logs: tail -n 100 logs/error.log
```

**Weekly Checklist (30 minutes):**
```bash
☐ Update dependencies: pip install --upgrade -r requirements.txt
☐ Run full test suite: pytest tests/
☐ Backup important data: ./scripts/backup.sh
☐ Review analytics trends
☐ Archive old files
☐ Clean Docker images: docker system prune
```

**Monthly Checklist (2 hours):**
```bash
☐ Review and optimize model performance
☐ Analyze A/B test results
☐ Update content strategy based on trends
☐ Review financial metrics and adjust pricing
☐ Update documentation
☐ System security audit
☐ Backup entire system
```

### 10.2 Model Updates

**When to Retrain:**
- Quality scores declining >10%
- New musical styles desired
- Accumulated >1000 human feedback samples
- Major dataset addition

**Retraining Procedure:**
```bash
# 1. Prepare new training data
python scripts/prepare_training_data.py \
    --input data/new_midi/ \
    --output data/processed/ \
    --augment

# 2. Train with curriculum learning
python scripts/train.py \
    --config configs/training.yaml \
    --curriculum \
    --epochs 100 \
    --checkpoint-dir checkpoints/

# 3. Evaluate
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --test-data data/test/

# 4. If better, deploy
if [ $? -eq 0 ]; then
    cp checkpoints/best_model.pt models/lofi_model.pt
    echo "Model updated successfully"
fi
```

### 10.3 System Backups

**Automated Daily Backup:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/lofi_$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup models
cp -r models/ $BACKUP_DIR/models/

# Backup configuration
cp -r configs/ $BACKUP_DIR/configs/
cp .env $BACKUP_DIR/.env

# Backup output (last 7 days)
find output/ -mtime -7 -type f -exec cp {} $BACKUP_DIR/output/ \;

# Backup database (if using)
# pg_dump lofi_db > $BACKUP_DIR/database.sql

# Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"

# Keep only last 30 days
find /backups/ -name "lofi_*.tar.gz" -mtime +30 -delete
```

**Schedule with cron:**
```bash
# Add to crontab
0 2 * * * /path/to/lofi/backup.sh
```

### 10.4 Security Updates

**Regular Security Checks:**
```bash
# Check for vulnerabilities
pip-audit

# Update security patches
pip install --upgrade pip setuptools

# Scan dependencies
safety check

# Check for exposed secrets
git secrets --scan-history
```

---

## 11. Quality Control

### 11.1 Quality Standards

**Minimum Standards for Release:**

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| Quality Score | 0.70 | 0.85+ | Overall AI score |
| LUFS | -15.0 | -14.0 ± 0.3 | Streaming standard |
| True Peak | < -0.1 dBFS | < -0.3 dBFS | Prevent clipping |
| Dynamic Range | > 6 dB | > 8 dB | Maintain dynamics |
| Stereo Correlation | 0.5-1.0 | 0.7-0.9 | Lofi sweet spot |
| Melodic Coherence | 0.70 | 0.85+ | Music theory check |
| Harmonic Variety | 0.60 | 0.75+ | Not too repetitive |

### 11.2 Quality Control Process

**Step 1: Automated Checks**
```python
def quality_control_pipeline(track_path, audio_path):
    """Run full QC pipeline."""

    checks = {
        'generation_quality': False,
        'lufs_target': False,
        'true_peak': False,
        'dynamic_range': False,
        'stereo_field': False
    }

    # 1. Check generation quality
    tokens = load_tokens(track_path)
    score = scorer.score_track(tokens)
    checks['generation_quality'] = score >= 0.70

    # 2. Check audio metrics
    audio, sr = sf.read(audio_path)
    loudness = LoudnessProcessor.measure_lufs(audio, sr)

    checks['lufs_target'] = abs(loudness['lufs_integrated'] - (-14.0)) < 1.0
    checks['true_peak'] = loudness['true_peak_db'] < -0.1

    # 3. Check dynamic range
    analysis = MasteringChain().analyze_master(audio, sr)
    checks['dynamic_range'] = analysis['dynamic_range_db'] > 6.0

    # 4. Check stereo field
    if audio.ndim == 2:
        stereo = StereoImaging.analyze_stereo_field(audio)
        checks['stereo_field'] = 0.5 < stereo['correlation'] < 1.0
    else:
        checks['stereo_field'] = True  # Mono is fine

    # Print results
    print("\n=== Quality Control ===")
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")

    all_passed = all(checks.values())
    return all_passed, checks

# Run QC
passed, details = quality_control_pipeline('track.mid', 'track.wav')

if passed:
    print("\n✅ APPROVED for distribution")
else:
    print("\n❌ REJECTED - needs adjustment")
```

**Step 2: Manual Review (Sampling)**
- Review 10% of output manually
- Focus on edge cases and low scores
- Collect feedback for RLHF training

**Step 3: User Feedback Loop**
```python
from src.rlhf import PreferenceCollector

collector = PreferenceCollector()

# Collect preferences
preference = collector.collect_preference(
    track_a='track_001.mid',
    track_b='track_002.mid',
    preferred='a',
    reason='Better melody, more coherent'
)

# Periodically retrain
if len(collector.preferences) >= 1000:
    # Train reward model
    reward_trainer = RewardModelTrainer(model)
    reward_trainer.train(collector.preferences)

    # Fine-tune with RLHF
    ppo_trainer = PPOTrainer(model, reward_trainer.reward_model)
    ppo_trainer.train()
```

---

## 12. Emergency Procedures

### 12.1 Service Outage

**If System Goes Down:**

1. **Check Docker services:**
   ```bash
   docker-compose ps
   docker-compose logs --tail=100
   ```

2. **Restart services:**
   ```bash
   docker-compose restart

   # Or full restart
   docker-compose down
   docker-compose up -d
   ```

3. **Check logs for errors:**
   ```bash
   tail -n 500 logs/error.log
   tail -n 500 logs/api.log
   ```

4. **Restore from backup if needed:**
   ```bash
   # Stop services
   docker-compose down

   # Restore backup
   tar -xzf /backups/lofi_20251117.tar.gz
   cp -r lofi_20251117/models/* models/
   cp lofi_20251117/.env .env

   # Restart
   docker-compose up -d
   ```

### 12.2 Data Loss

**If Important Data Lost:**

1. **Check backups:**
   ```bash
   ls -lh /backups/lofi_*.tar.gz | tail -n 7
   ```

2. **Restore specific data:**
   ```bash
   # Extract backup
   tar -xzf /backups/lofi_20251117.tar.gz

   # Restore needed files
   cp lofi_20251117/output/* output/
   cp lofi_20251117/models/* models/
   ```

3. **Verify restored data:**
   ```bash
   python scripts/verify_data.py
   ```

### 12.3 Model Degradation

**If Model Performance Suddenly Drops:**

1. **Revert to previous model:**
   ```bash
   cp models/lofi_model_backup.pt models/lofi_model.pt
   ```

2. **Verify previous performance:**
   ```bash
   python scripts/evaluate.py --model models/lofi_model.pt
   ```

3. **Investigate cause:**
   - Check recent training logs
   - Review last data changes
   - Verify model file integrity: `md5sum models/lofi_model.pt`

### 12.4 API Rate Limits

**If YouTube API Quota Exceeded:**

1. **Check quota status:**
   ```bash
   python scripts/check_api_quota.py
   ```

2. **Switch to backup account** (if configured):
   ```python
   uploader = YouTubeUploader(backup_credentials=True)
   ```

3. **Delay uploads until quota resets:**
   ```python
   # YouTube quota resets at midnight Pacific Time
   # Reschedule uploads
   for schedule in uploader.upload_queue:
       schedule.scheduled_time += timedelta(hours=12)
   ```

---

## Appendix A: Keyboard Shortcuts & CLI Commands

### Quick Commands

```bash
# Generate single track
python -m src.cli generate --tempo 75 --key Am --mood peaceful

# Batch generate
python -m src.cli batch-generate --count 50 --output output/

# Master track
python -m src.cli master --input track.wav --preset streaming

# Upload to YouTube
python -m src.cli upload --video track.mp4 --title "Chill Lofi"

# Generate daily report
python -m src.cli report --type daily

# Check system health
python -m src.cli health

# Run quality check
python -m src.cli qc --input output/

# Start API server
python -m src.api

# Start Jupyter
docker-compose up jupyter
```

---

## Appendix B: Configuration Templates

### Production config.yaml

```yaml
# Production Configuration
system:
  mode: "production"
  log_level: "INFO"
  parallel_workers: 4

generation:
  model_path: "models/lofi_model.pt"
  batch_size: 16
  max_length: 1024
  temperature: 0.9
  quality_threshold: 0.7

production:
  mastering:
    preset: "streaming"
    target_lufs: -14.0
    ceiling: -0.3
  sample_rate: 44100

automation:
  youtube:
    auto_upload: true
    upload_hour: 18
    create_playlists: true

  analytics:
    update_interval: 3600
    generate_daily_report: true

monitoring:
  prometheus:
    enabled: true
    port: 9090

  alerts:
    quality_drift_threshold: 0.1
    error_threshold: 10
```

---

## Appendix C: Glossary

**LUFS:** Loudness Units relative to Full Scale - standard for measuring loudness

**MIDI:** Musical Instrument Digital Interface - protocol for digital music

**True Peak:** The actual peak level of audio after digital-to-analog conversion

**Dynamic Range:** Difference between loudest and quietest parts (in dB)

**Mid-Side (M/S):** Stereo processing technique separating center and sides

**RLHF:** Reinforcement Learning from Human Feedback

**Diffusion Model:** Generative model that learns by reversing noise

**Style Transfer:** Applying style of one piece to content of another

**GPT-2:** Generative Pre-trained Transformer (language model architecture)

**Beam Search:** Decoding algorithm that explores multiple paths

**SATB:** Soprano, Alto, Tenor, Bass (four-part harmony)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | System | Initial SOP creation |

---

**END OF STANDARD OPERATING PROCEDURES**

For questions or updates, contact: [Your Contact Info]
