# 🛠️ Setup Guide - Lo-Fi Music Generator

Complete installation and setup instructions for the lo-fi music generator.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows 10+
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 20GB free space
- **Python**: 3.8 or higher

### Recommended (for Training)

- **GPU**: NVIDIA RTX 3090, 4090, or A100
- **VRAM**: 24GB+
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **CUDA**: 11.8 or higher

### Cloud Alternatives

If you don't have a GPU, consider:

- **Google Colab Pro** - $10/month, good GPUs
- **AWS EC2** - g4dn.xlarge or p3.2xlarge instances
- **Vast.ai** - Cheap GPU rentals
- **Lambda Labs** - ML-optimized cloud GPUs

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/lofi-music-generator.git
cd lofi-music-generator
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

**With GPU (CUDA 11.8):**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**CPU Only:**
```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 4. Install FluidSynth (Optional but Recommended)

FluidSynth provides better MIDI to audio conversion.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install fluidsynth
```

**macOS:**
```bash
brew install fluid-synth
```

**Windows:**
Download from [FluidSynth website](https://github.com/FluidSynth/fluidsynth/releases)

### 5. Download SoundFont

Download a General MIDI soundfont for audio rendering:

```bash
# Create soundfonts directory
mkdir -p soundfonts

# Download GeneralUser GS soundfont (recommended)
cd soundfonts
wget https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip
unzip GeneralUser_GS_1.471.zip
mv "GeneralUser GS 1.471/GeneralUser GS v1.471.sf2" GeneralUser_GS.sf2
cd ..
```

Or download manually from:
- [GeneralUser GS](http://www.schristiancollins.com/generaluser.php)
- [FluidR3_GM](https://member.keymusician.com/Member/FluidR3_GM/index.html)

Update `config.yaml` with the soundfont path:
```yaml
audio:
  soundfont_path: "soundfonts/GeneralUser_GS.sf2"
```

## Configuration

### 1. Review Configuration

Open `config.yaml` and review settings:

```yaml
# Key settings to check:
data:
  midi_dir: "data/midi"  # Where your MIDI files are

training:
  num_epochs: 50  # Adjust based on dataset size
  batch_size: 4   # Reduce if GPU memory issues
  fp16: true      # Disable if GPU doesn't support FP16

generation:
  num_tracks: 10  # For testing
```

### 2. Adjust for Your Hardware

**Low GPU Memory (<16GB):**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16  # Keep effective batch size at 32
```

**No GPU (CPU only):**
```yaml
training:
  device: "cpu"
  batch_size: 1
  fp16: false
  num_epochs: 10  # Reduce for reasonable time
```

**High-end GPU (24GB+):**
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
```

## Data Preparation

### 1. Create Data Directories

```bash
mkdir -p data/midi
mkdir -p data/tokens
mkdir -p data/datasets
mkdir -p models
mkdir -p output
```

### 2. Obtain MIDI Files

You need **lo-fi style MIDI files** for training. Sources:

**Free Sources:**
- [FreeMIDI.org](https://freemidi.org/) - Search "lo-fi", "chill", "jazz"
- [MIDI World](https://www.midiworld.com/) - Various genres
- [BitMidi](https://bitmidi.com/) - Free MIDI database

**Tips for Finding Good MIDI:**
- Look for: jazz, hip-hop, chill, ambient
- Tempo: 60-95 BPM
- Must have: drums/percussion
- Instruments: piano, bass, light synths

**Recommended Amount:**
- Minimum: 50 high-quality MIDI files
- Good: 200-500 MIDI files
- Optimal: 1000+ MIDI files

### 3. Add MIDI Files

Place all MIDI files in `data/midi/`:

```bash
# Example structure:
data/midi/
├── track001.mid
├── track002.mid
├── track003.mid
└── ...
```

### 4. Quality Check

The tokenizer automatically filters MIDI files based on:
- Tempo: 60-95 BPM (lo-fi range)
- Duration: 30-300 seconds
- Has drums: Required
- Note density: 0.5-8.0 notes/second

You can adjust these in `config.yaml`:

```yaml
data:
  quality_filters:
    min_tempo: 60
    max_tempo: 95
    require_drums: true
```

## Verification

### 1. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import miditok; print(f'MidiTok: {miditok.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True  (or False if CPU only)
Transformers: 4.x.x
MidiTok: 2.x.x
```

### 2. Test FluidSynth

```bash
fluidsynth --version
```

Should show FluidSynth version.

### 3. Test Pipeline

```bash
# Test tokenization (with your MIDI files)
python scripts/01_tokenize.py --help

# Should show script help without errors
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1** - Reduce batch size:
```yaml
training:
  batch_size: 2  # or even 1
  gradient_accumulation_steps: 16
```

**Solution 2** - Use gradient checkpointing (add to model.py):
```python
model.gradient_checkpointing_enable()
```

**Solution 3** - Use CPU:
```yaml
training:
  device: "cpu"
  fp16: false
```

### Issue: No MIDI Files Pass Quality Check

**Solution 1** - Disable quality check:
```bash
python scripts/01_tokenize.py --no-quality-check
```

**Solution 2** - Adjust filters in config.yaml:
```yaml
data:
  quality_filters:
    min_tempo: 50  # Lower minimum
    max_tempo: 120  # Higher maximum
    require_drums: false  # Don't require drums
```

### Issue: FluidSynth Not Found

**Solution** - The system will fall back to pretty_midi synthesis automatically. This works but may sound different.

To install FluidSynth:
- Ubuntu: `sudo apt-get install fluidsynth`
- macOS: `brew install fluid-synth`
- Windows: Download from GitHub releases

### Issue: Import Errors

**Solution** - Reinstall dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue: Slow Training on CPU

**Solution 1** - Reduce model size:
```yaml
model:
  num_layers: 6  # Half size = faster training
  embedding_dim: 512
```

**Solution 2** - Reduce epochs:
```yaml
training:
  num_epochs: 10  # Instead of 50
```

**Solution 3** - Use cloud GPU (recommended).

### Issue: Permission Denied on Scripts

**Solution**:
```bash
chmod +x scripts/*.py
```

## Next Steps

Once setup is complete:

1. ✅ Verify all dependencies installed
2. ✅ MIDI files in `data/midi/`
3. ✅ Configuration adjusted for your hardware
4. ✅ Test commands run without errors

**You're ready!** Proceed to [USAGE.md](USAGE.md) for the complete workflow.

## Additional Resources

### GPU Cloud Providers

- **Google Colab Pro**: https://colab.research.google.com/
- **Vast.ai**: https://vast.ai/ (cheap GPU rentals)
- **Lambda Labs**: https://lambdalabs.com/
- **AWS**: https://aws.amazon.com/ec2/instance-types/

### MIDI Resources

- **Free MIDI Files**: https://freemidi.org/
- **MIDI DB**: https://www.mididb.com/
- **Create Your Own**: MuseScore, FL Studio, Ableton

### Learning Resources

- **MidiTok Docs**: https://miditok.readthedocs.io/
- **HuggingFace Course**: https://huggingface.co/course/
- **Music AI**: https://magenta.tensorflow.org/

---

**Need help?** Open an issue on GitHub or check existing issues!
