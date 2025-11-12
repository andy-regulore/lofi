# 📖 Usage Guide - Lo-Fi Music Generator

Complete guide to using the lo-fi music generator system.

## Table of Contents

1. [Complete Workflow](#complete-workflow)
2. [Step-by-Step Guide](#step-by-step-guide)
3. [Advanced Usage](#advanced-usage)
4. [Configuration](#configuration)
5. [Tips & Best Practices](#tips--best-practices)
6. [YouTube Upload Guide](#youtube-upload-guide)

## Complete Workflow

The complete workflow consists of 5 steps:

```
MIDI Files → Tokenization → Dataset → Training → Generation → Lo-Fi Audio
   (01)         (02)         (03)       (04)          (05)
```

1. **Tokenize**: Convert MIDI files to token sequences
2. **Build Dataset**: Create training dataset with chunking
3. **Train**: Train 117M parameter GPT-2 model
4. **Generate**: Create new lo-fi tracks
5. **Batch Generate**: Produce 75-100 tracks for YouTube

## Step-by-Step Guide

### Step 1: Tokenize MIDI Files

Convert MIDI files to tokens with quality filtering.

**Basic Usage:**
```bash
python scripts/01_tokenize.py
```

**With Options:**
```bash
python scripts/01_tokenize.py \
  --midi-dir data/midi \
  --output-dir data/tokens \
  --config config.yaml
```

**Disable Quality Filtering:**
```bash
python scripts/01_tokenize.py --no-quality-check
```

**What It Does:**
- Scans `data/midi/` for MIDI files
- Checks quality (tempo, drums, density)
- Tokenizes with MidiTok REMI
- Saves to `data/tokens/`
- Creates metadata JSON

**Expected Output:**
```
Found 150 MIDI files
Vocabulary size: 20000
Quality filters:
  Tempo range: 60-95 BPM
  ...
Tokenization complete: 120/150 files passed quality check
```

**Troubleshooting:**
- If few files pass: adjust quality filters in `config.yaml`
- If no files found: check MIDI files are in `data/midi/`
- Use `--no-quality-check` for testing

---

### Step 2: Build Training Dataset

Create HuggingFace dataset from tokenized sequences.

**Basic Usage:**
```bash
python scripts/02_build_dataset.py
```

**With Options:**
```bash
python scripts/02_build_dataset.py \
  --tokens-dir data/tokens \
  --output-dir data/datasets \
  --eval-split 0.1
```

**What It Does:**
- Loads tokenized sequences
- Chunks into 1024-token samples
- Splits into train/eval (90/10)
- Creates HuggingFace dataset
- Saves to `data/datasets/`

**Expected Output:**
```
Loaded 120 sequences
Created 4800 training samples
Train samples: 4320
Eval samples: 480
Dataset saved to: data/datasets
```

---

### Step 3: Train Model

Train the 117M parameter GPT-2 model.

**Basic Usage:**
```bash
python scripts/03_train.py
```

**With Options:**
```bash
python scripts/03_train.py \
  --dataset-dir data/datasets \
  --output-dir models/lofi-gpt2 \
  --config config.yaml
```

**Resume Training:**
```bash
python scripts/03_train.py --resume
```

**What It Does:**
- Loads dataset
- Initializes 117M parameter GPT-2 model
- Trains for 50 epochs (or until early stopping)
- Saves checkpoints every 1000 steps
- Monitors eval loss
- Saves best model

**Expected Duration:**
- RTX 3090: 8-12 hours
- RTX 4090: 6-8 hours
- CPU: Several days (not recommended)

**Expected Output:**
```
Model architecture:
  Total parameters: 117,234,432 (117.2M)
  Vocabulary size: 20000
  ...
Training...
  [Progress bars and metrics]

Training complete!
  Final train loss: 2.134
  Final eval loss: 2.387
  ✓ Target eval loss (2.5) achieved!
```

**Monitoring Training:**

View TensorBoard logs:
```bash
tensorboard --logdir models/lofi-gpt2/logs
```

**Troubleshooting:**
- CUDA out of memory: reduce `batch_size` in config
- Slow training: check GPU is being used
- High loss: train longer or get more data

---

### Step 4: Generate Tracks

Generate individual tracks with control.

**Basic Usage:**
```bash
python scripts/04_generate.py --num-tracks 10
```

**With Specific Parameters:**
```bash
python scripts/04_generate.py \
  --num-tracks 5 \
  --tempo 75 \
  --key "Am" \
  --mood "chill" \
  --name "my_lofi" \
  --seed 42
```

**MIDI Only (Fast):**
```bash
python scripts/04_generate.py --num-tracks 10 --midi-only
```

**What It Does:**
- Loads trained model
- Generates MIDI with conditioning
- Converts to WAV
- Applies lo-fi effects
- Normalizes to -14 LUFS
- Saves to `output/`

**Output Files:**
```
output/
├── my_lofi_001.mid        # MIDI file
├── my_lofi_001_lofi.wav   # Lo-fi processed audio
└── my_lofi_metadata.json  # Track metadata
```

**Generation Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--tempo` | BPM (60-95) | `--tempo 72` |
| `--key` | Musical key | `--key "C"` or `--key "Am"` |
| `--mood` | Mood/vibe | `--mood "chill"` |
| `--seed` | Random seed | `--seed 42` |

**Available Moods:**
- `chill` - Relaxed, laid-back
- `melancholic` - Sad, emotional
- `upbeat` - Energetic, positive
- `relaxed` - Calm, peaceful
- `dreamy` - Atmospheric, ambient

**Expected Output:**
```
Generating track 1/10
  Tempo: 75.0, Key: Am, Mood: chill
  Generated 1847 tokens
  Quality score: 8.2/10
  Saved MIDI to output/my_lofi_001.mid
  Processing audio...
  Lo-fi audio saved: output/my_lofi_001_lofi.wav

Generation complete!
  Generated 10 tracks
  Output directory: output/
```

---

### Step 5: Batch Generate for YouTube

Generate 75-100 tracks for commercial use.

**Generate 100 Tracks:**
```bash
python scripts/05_batch_generate.py --num-tracks 100
```

**With Parallel Processing:**
```bash
python scripts/05_batch_generate.py \
  --num-tracks 100 \
  --parallel 4 \
  --min-quality 7.0 \
  --name "lofi_summer_2024"
```

**What It Does:**
- Generates specified number of tracks
- Ensures variety (different tempos, keys, moods)
- Applies quality filtering
- Processes all audio
- Creates organized output
- Generates batch metadata

**Output Structure:**
```
output/lofi_summer_2024/
├── lofi_summer_2024_001.mid
├── lofi_summer_2024_001_lofi.wav
├── lofi_summer_2024_002.mid
├── lofi_summer_2024_002_lofi.wav
├── ...
├── lofi_summer_2024_metadata.json
└── lofi_summer_2024_summary.json
```

**Expected Duration:**
- 100 tracks: 3-5 hours (with parallel=4)
- Per track: ~2-3 minutes

**Expected Output:**
```
Generating 100 tracks...
  [Progress bar]

Batch generation complete!
  Total tracks: 100
  Successful: 98
  With audio: 98
  Failed: 2
  Duration: 4.2 hours (2.5 min/track)
  High quality tracks (≥7.0): 87/98

Ready for YouTube upload!
  Generated 98 tracks ready for upload!
```

## Advanced Usage

### Custom Configuration

Create custom config for specific projects:

```bash
# Copy default config
cp config.yaml my_config.yaml

# Edit settings
nano my_config.yaml

# Use custom config
python scripts/03_train.py --config my_config.yaml
```

### Training on Custom Data

```yaml
# config.yaml
data:
  midi_dir: "my_midi_collection"
  quality_filters:
    min_tempo: 70
    max_tempo: 85
    require_drums: true
```

### Fine-Tuning Pre-trained Model

```bash
# Train first model
python scripts/03_train.py --output-dir models/base_model

# Fine-tune on new data
# (Add new MIDI files to data/midi/)
python scripts/01_tokenize.py
python scripts/02_build_dataset.py
python scripts/03_train.py --output-dir models/finetuned_model
```

### Generating Specific Styles

**Chill Study Beats:**
```bash
python scripts/04_generate.py \
  --tempo 70 \
  --mood "chill" \
  --num-tracks 20 \
  --name "study_beats"
```

**Melancholic Lo-Fi:**
```bash
python scripts/04_generate.py \
  --tempo 65 \
  --key "Am" \
  --mood "melancholic" \
  --num-tracks 20 \
  --name "sad_lofi"
```

**Upbeat Lo-Fi:**
```bash
python scripts/04_generate.py \
  --tempo 90 \
  --mood "upbeat" \
  --num-tracks 20 \
  --name "happy_beats"
```

## Configuration

### Key Config Sections

**Quality Filtering:**
```yaml
data:
  quality_filters:
    min_tempo: 60      # Minimum BPM
    max_tempo: 95      # Maximum BPM
    min_duration: 30   # Minimum seconds
    max_duration: 300  # Maximum seconds
    require_drums: true
    min_note_density: 0.5
    max_note_density: 8.0
```

**Training:**
```yaml
training:
  num_epochs: 50
  batch_size: 4
  learning_rate: 3.0e-4
  fp16: true  # Mixed precision
  early_stopping_patience: 5
  target_eval_loss: 2.5
```

**Generation:**
```yaml
generation:
  temperature: 0.9  # Higher = more random
  top_k: 50        # Limit to top 50 tokens
  top_p: 0.95      # Nucleus sampling
  num_tracks: 10
```

**Lo-Fi Effects:**
```yaml
audio:
  lofi_effects:
    downsample_rate: 22050
    lowpass_cutoff: 3500
    highpass_cutoff: 100
    bit_depth: 12
    vinyl_crackle:
      enabled: true
      intensity: 0.015
```

## Tips & Best Practices

### Data Collection

✅ **Do:**
- Use 200+ MIDI files for best results
- Focus on lo-fi, jazz, hip-hop styles
- Ensure variety (different keys, tempos)
- Use high-quality MIDI files

❌ **Don't:**
- Use copyrighted MIDI without permission
- Mix vastly different genres
- Use very short (<30s) or very long (>5min) files

### Training

✅ **Do:**
- Monitor TensorBoard logs
- Use GPU for training (8-12 hours vs days)
- Enable FP16 for faster training
- Let early stopping prevent overfitting

❌ **Don't:**
- Train on CPU (too slow)
- Stop training before eval loss < 3.0
- Use batch size > GPU memory allows

### Generation

✅ **Do:**
- Generate variety (different tempos, keys, moods)
- Use quality filtering
- Test with small batches first
- Keep successful generation parameters

❌ **Don't:**
- Generate all tracks with same parameters
- Skip audio processing
- Ignore quality scores

### Audio Quality

✅ **Do:**
- Use -14 LUFS normalization for YouTube
- Apply lo-fi effects consistently
- Test audio on different speakers
- Use FluidSynth if available

❌ **Don't:**
- Over-compress audio
- Skip normalization
- Use excessive effects

## YouTube Upload Guide

### 1. Prepare Content

**What You Need:**
- 75-100 lo-fi audio tracks (.wav)
- Background visuals (static images or animations)
- Channel art and banner
- Descriptions and tags

**Tools for Visuals:**
- Canva - Easy graphic design
- DALL-E / Midjourney - AI-generated art
- Unsplash - Free stock photos
- After Effects - Animated backgrounds

### 2. Create Videos

**Option 1 - Simple (Static Image):**
```bash
# Use FFmpeg to create video from audio + image
ffmpeg -loop 1 -i background.jpg -i track_001_lofi.wav \
  -c:v libx264 -c:a aac -b:a 192k -shortest output_001.mp4
```

**Option 2 - Automated:**
Use tools like:
- [Headliner](https://www.headliner.app/)
- [Kapwing](https://www.kapwing.com/)
- Python + MoviePy (automated batch processing)

### 3. Optimize for YouTube

**Title Format:**
```
Chill Lo-Fi Beats to Study/Relax 📚 [1 Hour] | Homework Music
Lo-Fi Hip Hop Radio 24/7 🎧 Beats to Sleep/Chill To
```

**Description Template:**
```
🎵 Chill lo-fi beats perfect for studying, working, or relaxing

Follow us for more lo-fi music!
👉 Subscribe: [Your Channel]
👉 Spotify: [Your Spotify]
👉 Instagram: [Your Instagram]

#lofi #chillbeats #studymusic #lofimusic #chillhop

---
🎧 Tracklist:
00:00 - Midnight Dreams
03:45 - Rainy Day Study
...

All music composed and produced by [Your Name]
© [Year] All Rights Reserved
```

**Tags:**
```
lofi, lo-fi, chill beats, study music, relaxing music,
homework music, lofi hip hop, chill hop, study beats,
focus music, ambient music, instrumental music
```

### 4. Upload Strategy

**Consistency:**
- Upload 2-3 tracks per week
- Build library to 50+ videos
- Create playlists by mood

**Engagement:**
- Respond to comments
- Create community posts
- Collaborate with other creators

**Monetization Timeline:**
- Week 1-4: Build initial library (20+ videos)
- Month 2-3: Reach 1000 subscribers
- Month 3-6: Reach 4000 watch hours
- Month 6+: Enable monetization

### 5. Revenue Optimization

**Multiple Platforms:**
- YouTube - Ad revenue
- Spotify - Streaming royalties
- Bandcamp - Direct sales
- Patreon - Fan support

**Expected Revenue (YouTube):**
- $1-3 per 1000 views (RPM)
- 10k views/month = $10-30
- 100k views/month = $100-300
- 500k views/month = $500-1500

---

## Need Help?

- Check [SETUP.md](SETUP.md) for installation issues
- Open GitHub issue for bugs
- Join Discord community (if available)

## Ready to Generate?

Start with Step 1 and work through the complete workflow. Good luck with your lo-fi music empire! 🎵💰
