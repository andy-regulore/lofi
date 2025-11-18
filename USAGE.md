# 🎵 Ultra-Pro Lo-Fi Music Generator - Complete Usage Guide

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/andy-regulore/lofi.git
cd lofi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Basic Generation (3 lines of code!)

```python
from src.generator import LoFiGenerator
from src.model import ConditionedLoFiModel
from src.tokenizer import LoFiTokenizer
import yaml

# Load config and create generator
with open('config.yaml') as f:
    config = yaml.safe_load(f)

tokenizer = LoFiTokenizer(config)
model = ConditionedLoFiModel(config, tokenizer.get_vocab_size())
generator = LoFiGenerator(model, tokenizer, config)

# Generate a track!
tokens, metadata = generator.generate_track(
    tempo=75,
    key='Am',
    mood='melancholic'
)
generator.tokens_to_midi(tokens, 'my_lofi_track.mid')
```

---

## 📚 Complete Feature Guide

### 1. **Basic Generation**

```python
# Generate with specific parameters
tokens, metadata = generator.generate_track(
    tempo=72,              # BPM (60-95 ideal for lo-fi)
    key='C',              # Musical key
    mood='chill',         # Mood: chill, melancholic, upbeat, relaxed, dreamy
    max_length=1024,      # Number of tokens
    temperature=0.9,      # Sampling temperature (0.7-1.2)
    top_k=50,            # Top-k sampling
    top_p=0.95,          # Nucleus sampling
    seed=42              # For reproducibility
)
```

### 2. **Quality Scoring** ⭐ NEW!

```python
from src.quality_scorer import MusicQualityScorer

scorer = MusicQualityScorer(config)

# Score MIDI tokens
quality = scorer.score_midi_tokens(tokens, metadata)
print(f"Quality: {quality:.2f}/10")

# Score audio file
audio_quality = scorer.score_audio('track.wav')
print(f"Audio quality: {audio_quality:.2f}/10")
```

**Quality metrics analyzed:**
- Token diversity and distribution
- Repetition patterns
- Tempo appropriateness
- Dynamic range
- Spectral features
- Frequency balance

### 3. **Batch Generation**

```python
# Generate multiple tracks with variety
metadata_list = generator.batch_generate(
    num_tracks=10,
    output_dir='output/batch',
    name_prefix='lofi_track',
    ensure_variety=True  # Ensures tempo/key/mood variety
)

# Filter by quality
from src.quality_scorer import MusicQualityScorer
scorer = MusicQualityScorer(config)

high_quality_tracks = []
for meta in metadata_list:
    if 'output_path' in meta:
        tokens = generator.tokenizer.tokenize_midi(meta['output_path'])
        if tokens:
            score = scorer.score_midi_tokens(tokens['tokens'], meta)
            if score >= 7.0:
                high_quality_tracks.append(meta)

print(f"Generated {len(high_quality_tracks)} high-quality tracks")
```

### 4. **Advanced ML Features** 🔥 ULTRA-PRO!

#### Style Transfer

```python
from src.advanced_ml import StyleTransfer

style_transfer = StyleTransfer(model, tokenizer)

# Extract style from reference track
reference_tokens = tokenizer.tokenize_midi('reference.mid')['tokens']
style_vector = style_transfer.extract_style_vector(reference_tokens)

# Generate new track with that style
new_tokens = style_transfer.generate_with_style(
    style_vector,
    num_tokens=1024,
    style_strength=0.7  # 0-1, higher = more similar
)
```

#### Track Variations

```python
from src.advanced_ml import TrackVariationGenerator

variation_gen = TrackVariationGenerator(model, tokenizer)

# Generate subtle variation
variation = variation_gen.generate_variation(
    original_tokens,
    variation_type='subtle'  # 'subtle', 'moderate', 'dramatic'
)

# Create remix by blending two tracks
remix = variation_gen.generate_remix(
    track_a_tokens,
    track_b_tokens,
    blend_ratio=0.5  # 0=all A, 1=all B
)
```

#### Gradient-Based Control

```python
from src.advanced_ml import GradientBasedController

controller = GradientBasedController(model, device='cuda')

# Generate with target characteristics
tokens = controller.generate_with_target(
    input_ids,
    target_characteristics={
        'diversity': 0.8,      # High diversity
        'rhythm_strength': 0.6  # Moderate rhythm
    },
    num_steps=50,
    learning_rate=0.01
)
```

#### Attention Visualization

```python
from src.advanced_ml import AttentionVisualizer

visualizer = AttentionVisualizer(model)
hooks = visualizer.register_hooks()

# Generate (attention will be captured)
tokens, _ = generator.generate_track(tempo=75)

# Visualize attention patterns
visualizer.visualize_attention(
    tokens,
    save_path='attention_heatmap.png'
)
```

### 5. **Music Theory Integration** 🎼 ULTRA-PRO!

```python
from src.music_theory import MusicTheoryEngine, RhythmEngine

# Initialize engines
theory = MusicTheoryEngine()
rhythm = RhythmEngine()

# Validate and correct melody
notes = [60, 62, 64, 65, 67]  # MIDI notes
is_valid, corrected = theory.validate_melody(notes, key='C', allow_chromatic=False)

# Get chord progression suggestions
chords = theory.suggest_chord_progression(
    key='Am',
    num_chords=4,
    style='lofi'
)
print(f"Suggested progression: {chords}")

# Analyze harmony
analysis = theory.analyze_harmony(notes)
print(f"Detected key: {analysis['likely_key']}")
print(f"Consonance: {analysis['consonance_score']:.2f}")

# Generate rhythm pattern
pattern = rhythm.generate_rhythm_pattern(
    time_signature='4/4',
    complexity=0.6,
    swing=0.3
)
```

### 6. **AI Audio Mastering** 🎚️ ULTRA-PRO!

```python
from src.audio_mastering import AIAudioMaster
import soundfile as sf

# Load audio
audio, sr = sf.read('track.wav')

# Auto-master
master = AIAudioMaster(sample_rate=sr)
mastered, info = master.auto_master(
    audio,
    target_loudness=-14.0,  # LUFS (Spotify standard)
    target_style='lofi'     # 'lofi', 'clean', or 'aggressive'
)

# Save
sf.write('track_mastered.wav', mastered.T, sr)

# View processing info
print(f"EQ applied: {info['eq']['eq_curve']}")
print(f"Compression: {info['compression']['num_bands']} bands")
print(f"Gain applied: {info['limiter']['gain_applied_db']:.2f} dB")
```

### 7. **Complete Audio Pipeline**

```python
from src.audio_processor import LoFiAudioProcessor

processor = LoFiAudioProcessor(config)

# Full pipeline: MIDI → WAV → Lo-Fi Effects
result = processor.process_midi_to_lofi(
    'generated.mid',
    'output/audio',
    name='my_track',
    save_clean=True,   # Save clean version
    save_lofi=True     # Save with lo-fi effects
)

print(f"Clean: {result.get('clean_wav_path')}")
print(f"Lo-fi: {result.get('lofi_wav_path')}")
```

### 8. **Resource Management**

```python
from src.utils.resource_manager import ResourceManager

rm = ResourceManager(
    min_free_disk_gb=10.0,
    min_free_memory_gb=2.0,
    gpu_memory_fraction=0.9
)

# Check all resources
resources = rm.check_all_resources()

if not resources['all_ok']:
    print("Warning: Low resources!")
    print(f"Disk: {resources['disk']['info']['free_gb']:.1f} GB")
    print(f"Memory: {resources['memory']['info']['available_gb']:.1f} GB")

# Get optimal device
device = rm.get_optimal_device()  # 'cuda', 'mps', or 'cpu'

# Clear GPU cache
rm.clear_gpu_cache()

# Cleanup old files
deleted = rm.cleanup_directory(
    'output/',
    pattern='*.mid',
    max_files=100  # Keep only 100 most recent
)
```

### 9. **Configuration Validation**

```python
from src.config_schema import LoFiConfig

# Load and validate config
config = LoFiConfig.from_yaml('config.yaml')

# Access validated settings
print(f"Model layers: {config.model.num_layers}")
print(f"Batch size: {config.training.batch_size}")

# Validate paths
warnings = config.validate_paths()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")

# Save validated config
config.to_yaml('config_validated.yaml')
```

### 10. **Secure Operations**

```python
from src.utils.security import SecurePathHandler, InputValidator

# Safe file operations
handler = SecurePathHandler(allowed_base_paths=['./data', './output'])

try:
    safe_path = handler.validate_path('output/track.mid')
    print(f"Safe path: {safe_path}")
except ValueError as e:
    print(f"Unsafe path: {e}")

# Input validation
validator = InputValidator()

tempo = validator.validate_tempo(75)       # OK
key = validator.validate_key('Am')         # OK
temp = validator.validate_temperature(0.9) # OK

# These would raise ValueError:
# validator.validate_tempo(500)  # Too high
# validator.validate_key('invalid')  # Invalid key
```

---

## 🐳 Docker Usage

### Development

```bash
# Start development container
docker-compose up dev

# Run tests
docker-compose run --rm dev pytest

# Run Jupyter notebooks
docker-compose up jupyter
# Access at http://localhost:8888
```

### Production Generation

```bash
# Generate tracks
docker-compose run --rm generate

# Batch generate (set number of tracks)
NUM_TRACKS=50 docker-compose run --rm batch-generate

# Monitor with TensorBoard
docker-compose up tensorboard
# Access at http://localhost:6006
```

### Training (GPU)

```bash
# Train model on GPU
docker-compose run --rm train

# With custom GPU
CUDA_VISIBLE_DEVICES=0 docker-compose run --rm train
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m "not slow"        # Skip slow tests

# Run tests in parallel
pytest -n auto

# Run with detailed output
pytest -v --tb=short
```

---

## 🔍 Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
flake8 src tests

# Type check
mypy src

# Security scan
bandit -r src

# Run all checks
pre-commit run --all-files
```

---

## 📊 Monitoring & Profiling

### TensorBoard

```bash
tensorboard --logdir models/lofi-gpt2/logs
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile generation
profiler = cProfile.Profile()
profiler.enable()

# Your code here
tokens, _ = generator.generate_track(tempo=75)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## 🎯 Production Deployment

### Environment Variables

Create `.env` file:

```bash
CUDA_VISIBLE_DEVICES=0
NUM_TRACKS=100
OUTPUT_DIR=./output/production
MODEL_PATH=./models/lofi-gpt2
LOG_LEVEL=INFO
```

### Batch Production Script

```python
#!/usr/bin/env python
"""Production batch generation script."""

import os
from pathlib import Path
import yaml
from src.generator import LoFiGenerator
from src.model import ConditionedLoFiModel
from src.tokenizer import LoFiTokenizer
from src.audio_processor import LoFiAudioProcessor
from src.quality_scorer import MusicQualityScorer
from src.utils.resource_manager import ResourceManager

def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Check resources
    rm = ResourceManager()
    resources = rm.check_all_resources()

    if not resources['all_ok']:
        print("WARNING: Low resources!")

    # Initialize components
    tokenizer = LoFiTokenizer(config)
    model = ConditionedLoFiModel(config, tokenizer.get_vocab_size())
    model.load(os.getenv('MODEL_PATH', 'models/lofi-gpt2'))

    generator = LoFiGenerator(model, tokenizer, config,
                            device=rm.get_optimal_device())
    processor = LoFiAudioProcessor(config)
    scorer = MusicQualityScorer(config)

    # Generate tracks
    num_tracks = int(os.getenv('NUM_TRACKS', 100))
    output_dir = Path(os.getenv('OUTPUT_DIR', 'output/production'))

    print(f"Generating {num_tracks} tracks...")

    metadata_list = generator.batch_generate(
        num_tracks=num_tracks,
        output_dir=str(output_dir / 'midi'),
        ensure_variety=True
    )

    # Process and filter by quality
    high_quality = 0

    for meta in metadata_list:
        if 'output_path' not in meta:
            continue

        # Score quality
        tokens = tokenizer.tokenize_midi(meta['output_path'])
        if not tokens:
            continue

        quality = scorer.score_midi_tokens(tokens['tokens'], meta)

        if quality >= 7.0:
            # Process to audio
            audio_result = processor.process_midi_to_lofi(
                meta['output_path'],
                str(output_dir / 'audio'),
                save_lofi=True,
                save_clean=False
            )

            if 'lofi_wav_path' in audio_result:
                high_quality += 1
                print(f"✓ Track {meta['track_number']}: Quality {quality:.2f}/10")

    print(f"\nCompleted: {high_quality}/{num_tracks} high-quality tracks")

    # Cleanup
    rm.cleanup_directory(output_dir / 'midi', pattern='*.mid', max_files=200)

if __name__ == '__main__':
    main()
```

---

## 🔧 Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size
config['training']['batch_size'] = 2

# Clear cache between generations
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing
config['model']['use_gradient_checkpointing'] = True
```

### Slow Generation

```python
# Use smaller model
config['model']['num_layers'] = 6  # Instead of 12

# Reduce max_length
tokens, _ = generator.generate_track(max_length=512)

# Use lower precision
config['training']['fp16'] = True
```

### Audio Quality Issues

```python
# Adjust lo-fi effects
config['audio']['lofi_effects']['lowpass_cutoff'] = 4000  # Higher
config['audio']['lofi_effects']['vinyl_crackle']['intensity'] = 0.01  # Lower
config['audio']['lofi_effects']['bit_depth'] = 14  # Higher

# Use AI mastering
from src.audio_mastering import AIAudioMaster
master = AIAudioMaster()
mastered, _ = master.auto_master(audio, target_style='clean')
```

---

## 📖 API Reference

Full API documentation: [link to Sphinx docs]

---

## 🎓 Advanced Topics

### Custom Training

```bash
# Prepare your MIDI data
python scripts/01_tokenize.py

# Build dataset
python scripts/02_build_dataset.py

# Train
python scripts/03_train.py

# Monitor training
tensorboard --logdir models/lofi-gpt2/logs
```

### Custom Models

```python
from src.model import LoFiMusicModel

# Create custom architecture
config['model']['num_layers'] = 24  # Larger model
config['model']['embedding_dim'] = 1024

model = LoFiMusicModel(config, vocab_size)
```

---

## 💡 Tips & Best Practices

1. **Start with small batches** - Test with 5-10 tracks before scaling
2. **Monitor quality scores** - Only process tracks scoring 7+/10
3. **Use resource management** - Check disk space before big batches
4. **Enable logging** - Set `LOG_LEVEL=DEBUG` for detailed output
5. **Backup models** - Save checkpoints regularly during training
6. **Version your configs** - Track config changes with git
7. **Use Docker for reproducibility** - Containers ensure consistent environment
8. **Profile performance** - Identify bottlenecks with cProfile
9. **Test on CPU first** - Verify logic before using expensive GPU time
10. **Clean up regularly** - Remove old outputs to save disk space

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.
