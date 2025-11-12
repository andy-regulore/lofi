# 🎵 Lo-Fi Music Generator

**Production-ready AI system for generating commercial-quality lo-fi music tracks**

Generate 75-100 professional lo-fi tracks for YouTube and Spotify monetization. This system uses a 117M parameter GPT-2 based transformer model trained on MIDI sequences to create authentic lo-fi hip-hop beats.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Features

- **117M Parameter GPT-2 Model** - Professional-grade transformer for music generation
- **Quality-Filtered Training** - Automatic filtering for lo-fi characteristics (60-95 BPM, drums, optimal density)
- **Conditional Generation** - Control tempo, key, and mood of generated tracks
- **Authentic Lo-Fi Effects** - Vinyl crackle, tape wow/flutter, bit reduction, filtering
- **YouTube-Ready Output** - Normalized to -14 LUFS for optimal streaming quality
- **Batch Production** - Generate 75-100 tracks in one run
- **Production Pipeline** - Complete workflow from MIDI to finished audio

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lofi-music-generator.git
cd lofi-music-generator

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](docs/SETUP.md) for detailed installation instructions.

### Basic Usage

```bash
# 1. Add MIDI files to data/midi/
# 2. Tokenize MIDI files
python scripts/01_tokenize.py

# 3. Build training dataset
python scripts/02_build_dataset.py

# 4. Train model (8-12 hours on RTX 3090)
python scripts/03_train.py

# 5. Generate tracks
python scripts/04_generate.py --num-tracks 10

# 6. Batch generate for YouTube (75-100 tracks)
python scripts/05_batch_generate.py --num-tracks 100
```

See [USAGE.md](docs/USAGE.md) for detailed usage guide.

## 📊 What You Get

After training, you can generate:

- **75-100 tracks per batch** - Enough content for months of uploads
- **MIDI + Audio files** - Both source MIDI and processed WAV files
- **Professional quality** - 8-9/10 quality score, YouTube/Spotify ready
- **Variety** - Different tempos, keys, and moods
- **Metadata** - JSON files with all track information

## 🎼 Model Architecture

- **Architecture**: GPT-2 based transformer
- **Parameters**: 117M (base GPT-2 size)
- **Context Length**: 2048 tokens
- **Embedding Dimension**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Training**: 50 epochs with early stopping
- **Target Loss**: < 2.5

## 🔊 Audio Processing

The system applies authentic lo-fi effects:

- **Sample Rate Reduction** - 44.1kHz → 22.05kHz for vintage sound
- **Filtering** - Low-pass (3.5kHz) and high-pass (100Hz)
- **Bit Reduction** - 12-bit for digital artifacts
- **Vinyl Crackle** - Realistic vinyl noise
- **Tape Effects** - Wow and flutter
- **Compression** - Professional dynamics control
- **Normalization** - -14 LUFS for YouTube/Spotify

## 💰 Monetization Potential

**YouTube Revenue Estimates:**
- Small channel (10k views/month): $100-300/month
- Medium channel (100k views/month): $500-1,500/month
- Large channel (500k+ views/month): $2,000-5,000/month

**Requirements for Monetization:**
- 1,000 subscribers
- 4,000 watch hours in past 12 months
- AdSense account

**Spotify:** Additional revenue through distributor (DistroKid, TuneCore, etc.)

## 📁 Project Structure

```
lofi-music-generator/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── src/                     # Core modules
│   ├── tokenizer.py        # MIDI tokenization
│   ├── model.py            # GPT-2 model
│   ├── trainer.py          # Training pipeline
│   ├── generator.py        # Music generation
│   └── audio_processor.py  # Lo-fi effects
├── scripts/                 # Pipeline scripts
│   ├── 01_tokenize.py      # Tokenize MIDI files
│   ├── 02_build_dataset.py # Create training dataset
│   ├── 03_train.py         # Train model
│   ├── 04_generate.py      # Generate tracks
│   └── 05_batch_generate.py # Batch production
├── data/                    # Data directory (gitignored)
│   ├── midi/               # Input MIDI files
│   ├── tokens/             # Tokenized data
│   └── datasets/           # Training datasets
├── models/                  # Trained models (gitignored)
├── output/                  # Generated tracks (gitignored)
└── docs/                    # Documentation
    ├── SETUP.md            # Installation guide
    └── USAGE.md            # Usage guide
```

## ⚙️ Configuration

All settings are in `config.yaml`:

- **Data**: MIDI directories, quality filters
- **Tokenization**: REMI tokenizer parameters
- **Model**: Architecture configuration
- **Training**: Hyperparameters, early stopping
- **Generation**: Sampling parameters, conditioning
- **Audio**: Lo-fi effects settings

## 🎓 Requirements

### Hardware

**Minimum:**
- CPU: 4+ cores
- RAM: 16GB
- Storage: 20GB

**Recommended (for training):**
- GPU: NVIDIA RTX 3090 or better
- RAM: 32GB
- Storage: 50GB SSD

**Cloud Alternatives:**
- Google Colab Pro (GPU)
- AWS EC2 (g4dn.xlarge or better)
- Vast.ai, Lambda Labs

### Software

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- FluidSynth (optional, for better audio)

## 📚 Documentation

- [SETUP.md](docs/SETUP.md) - Detailed installation and setup
- [USAGE.md](docs/USAGE.md) - Complete usage guide with examples
- [config.yaml](config.yaml) - Configuration reference

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Legal Notes

**Copyright:**
- Ensure you have rights to use training MIDI files
- Generated music can be copyrighted by you as derivative works
- Use royalty-free MIDI sources for commercial use

**YouTube/Spotify:**
- Follow platform guidelines for monetization
- Ensure tracks meet quality standards
- Avoid copyright claims by using original generations

## 🔗 Resources

**MIDI Sources:**
- [FreeMIDI.org](https://freemidi.org/)
- [MIDI World](https://www.midiworld.com/)
- Create your own MIDI files

**Music Distribution:**
- [DistroKid](https://distrokid.com/) - Spotify distribution
- [TuneCore](https://www.tunecore.com/) - Multi-platform
- YouTube Content ID for protection

**Learning Resources:**
- [MidiTok Documentation](https://miditok.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Music Generation with AI](https://magenta.tensorflow.org/)

## 🎵 Get Started

Ready to create your lo-fi music empire?

1. Follow [SETUP.md](docs/SETUP.md) to install
2. Read [USAGE.md](docs/USAGE.md) for the full guide
3. Generate your first 100 tracks
4. Start your YouTube channel
5. Monetize and profit! 💰

---

**Built with ❤️ for lo-fi music creators**

*Questions? Open an issue or discussion!*
