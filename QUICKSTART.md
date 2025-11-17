# ⚡ Quick Start Guide - LoFi Music Empire

Get started in **5 minutes**!

---

## 🎯 Goal

Generate your first LoFi track with video, metadata, and thumbnail.

---

## 📋 Prerequisites

- Python 3.9+
- 10GB free disk space
- Internet connection

---

## 🚀 Steps

### 1. Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Create output directories
mkdir -p output/{audio,videos,thumbnails,metadata}
```

### 2. Start Web UI (30 seconds)

```bash
# Start API server
python api_server.py
```

Open browser to: **http://localhost:8000**

### 3. Generate Your First Track (2 minutes)

**Option A: Web UI** (Recommended)

1. Click **"🎼 Generate"** tab
2. Select mood: **Chill**
3. Duration: **180 seconds**
4. Count: **1**
5. Click **"🎵 Generate Music"**
6. Wait 10-30 seconds
7. See track appear in list below

**Option B: Command Line**

```bash
python orchestrator.py --mode single --mood chill --duration 180
```

### 4. Check Output

```bash
ls -la output/audio/      # Generated audio
ls -la output/videos/     # Generated video
ls -la output/thumbnails/ # Generated thumbnail
ls -la output/metadata/   # Generated metadata
```

---

## 🎨 Try Different Moods

```bash
# Focus beats
python orchestrator.py --mode single --mood focus

# Happy vibes
python orchestrator.py --mode single --mood happy

# Peaceful ambience
python orchestrator.py --mode single --mood peaceful
```

---

## 📦 Batch Generation

Generate 5 tracks at once:

```bash
python orchestrator.py --mode batch --count 5 --mood chill
```

---

## 🎬 Video Templates

Test different video styles:

**Via Web UI**:
1. Go to **"🎬 Videos"** tab
2. Enter audio path from previous generation
3. Try templates:
   - `classic_lofi` - Circular visualizer + particles
   - `modern_spectrum` - Spectrum bars
   - `cyberpunk_wave` - Grid + glow
   - `minimal_bars` - Clean bars
   - `vintage_vinyl` - Retro style

**Via Code**:

```python
from src.video_generator import VideoGenerator, TemplateLibrary

generator = VideoGenerator()

generator.generate_video(
    audio_path="output/audio/track_123.wav",
    output_path="output/videos/track_123.mp4",
    template=TemplateLibrary.get_template('cyberpunk_wave'),
    title="Cyberpunk Chill Beats",
    artist="LoFi AI"
)
```

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "metadata": {
    "artist_name": "YOUR NAME HERE"
  },
  "video": {
    "default_template": "modern_spectrum"
  },
  "generation": {
    "default_mood": "focus"
  }
}
```

---

## 🔌 Integrate Your Model

**IMPORTANT**: The system uses a placeholder for generation. Integrate your private music model:

1. Open `orchestrator.py`
2. Find `generate_music()` method (line ~150)
3. Replace placeholder with your model:

```python
def generate_music(self, mood: str, duration: int, ...):
    # Import your model
    from your_model import generate

    # Generate audio
    audio_path = generate(mood=mood, duration=duration)

    # Return track info
    return {
        'audio_path': audio_path,
        'melody_notes': [...],
        'chords': [...],
        ...
    }
```

4. Test:
```bash
python orchestrator.py --mode single
```

---

## 🔄 Daily Automation

Set up daily content generation:

```bash
# Run daily workflow (generates N tracks based on schedule)
python orchestrator.py --mode daily
```

Add to crontab (Linux/Mac):
```bash
crontab -e

# Add line (runs daily at 3 AM):
0 3 * * * cd /path/to/lofi && python orchestrator.py --mode daily
```

---

## 📊 Monitor Progress

### Web UI Dashboard

- **Real-time stats**: Track totals
- **Job queue**: Monitor running jobs
- **Analytics**: View insights

### API Endpoints

```bash
# System status
curl http://localhost:8000/api/status

# List tracks
curl http://localhost:8000/api/tracks

# List jobs
curl http://localhost:8000/api/jobs
```

---

## 🎯 Next Steps

1. **Customize**: Edit `config.json` for your preferences
2. **Integrate Model**: Add your music generation model
3. **Schedule Content**: Set up content calendar
4. **YouTube Setup**: Configure YouTube API for uploads
5. **Automate**: Set up daily workflow with cron
6. **Scale**: Use batch mode for bulk generation

---

## 🆘 Common Issues

### Port 8000 in use?
```bash
# Use different port
python -c "from api_server import app; import uvicorn; uvicorn.run(app, port=8080)"
```

### Dependencies missing?
```bash
pip install --upgrade -r requirements.txt
```

### Permission errors?
```bash
chmod -R 755 output/
```

---

## 📚 Full Documentation

- **Complete Setup**: See `SETUP_GUIDE.md`
- **System Manual**: See `SOP.md`
- **API Docs**: http://localhost:8000/docs

---

## ✅ Success!

You now have:
- ✅ Working web dashboard
- ✅ Complete automation pipeline
- ✅ Generated sample content
- ✅ Understanding of workflow

**Ready to scale to 100k+ subscribers!** 🚀

---

**Total Time**: ~5 minutes
**Next**: Integrate your music model and start creating!
