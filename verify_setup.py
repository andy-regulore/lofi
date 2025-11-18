#!/usr/bin/env python3
"""
Setup Verification Script
Checks if the LoFi system is properly configured
"""

import sys
from pathlib import Path
import subprocess

def check_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_pass(msg):
    print(f"✅ {msg}")

def check_fail(msg):
    print(f"❌ {msg}")

def check_warn(msg):
    print(f"⚠️  {msg}")

def main():
    print("\n" + "🎵"*30)
    print("     LoFi Music Generator - Setup Verification")
    print("🎵"*30)

    # Check 1: Directory Structure
    check_section("1. Directory Structure")

    required_dirs = [
        'data/training',
        'models',
        'output/audio',
        'output/metadata',
        'src',
        'templates'
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            check_pass(f"{dir_path}/ exists")
        else:
            check_fail(f"{dir_path}/ does NOT exist")
            print(f"   → Create with: mkdir -p {dir_path}")

    # Check 2: Training Files
    check_section("2. Training Data Files")

    training_dir = Path('data/training')
    if not training_dir.exists():
        check_fail("data/training/ directory does not exist!")
        print("   → Create with: mkdir -p data/training")
        print("   → Then copy your MIDI files there")
    else:
        midi_files = list(training_dir.glob('*.mid')) + list(training_dir.glob('*.midi'))
        wav_files = list(training_dir.glob('*.wav'))

        if len(midi_files) > 0:
            check_pass(f"Found {len(midi_files):,} MIDI files")
            print(f"   → These will be used for training")
        else:
            check_fail("No MIDI files found in data/training/")
            print("   → Copy your .mid or .midi files to data/training/")

        if len(wav_files) > 0:
            check_warn(f"Found {len(wav_files):,} WAV files")
            print(f"   → WAV files are NOT yet supported for training")
            print(f"   → Only MIDI files will be used")

    # Check 3: Required Files
    check_section("3. Required Files")

    required_files = [
        'config.yaml',
        'web_ui.py',
        'api_server.py',
        'requirements.txt',
        'src/trainer.py',
        'src/tokenizer.py',
        'src/model.py'
    ]

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            check_pass(f"{file_path} exists")
        else:
            check_fail(f"{file_path} does NOT exist")

    # Check 4: Python Packages
    check_section("4. Python Dependencies")

    critical_packages = [
        'torch',
        'transformers',
        'flask',
        'numpy',
        'soundfile',
        'yaml',
        'sklearn'
    ]

    for package in critical_packages:
        try:
            __import__(package)
            check_pass(f"{package} installed")
        except ImportError:
            check_fail(f"{package} NOT installed")
            print(f"   → Install with: pip install -r requirements.txt")

    # Check 5: GPU Availability
    check_section("5. Hardware (GPU)")

    try:
        import torch
        if torch.cuda.is_available():
            check_pass(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   → Training will be FAST (GPU-accelerated)")
        else:
            check_warn("No CUDA GPU detected")
            print(f"   → Training will run on CPU (MUCH slower)")
            print(f"   → Expect 50-100x longer training times")
    except Exception as e:
        check_fail(f"Could not check GPU: {e}")

    # Check 6: Git Status
    check_section("6. Git Status")

    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        check_pass(f"Current branch: {branch}")

        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():
            check_warn("You have uncommitted changes")
        else:
            check_pass("Working directory is clean")

    except Exception as e:
        check_warn(f"Could not check git status: {e}")

    # Summary
    check_section("SUMMARY & NEXT STEPS")

    print("\n📝 To start the Web UI:")
    print("   python web_ui.py")
    print("\n📝 To start the API server:")
    print("   uvicorn api_server:app --reload")
    print("\n📝 To check training file count:")
    print("   ls -la data/training/*.mid* | wc -l")
    print("\n📝 To test with small dataset first:")
    print("   # Copy only 100 files for testing")
    print("   mkdir -p data/training_test")
    print("   cp /path/to/your/midi/files/*.mid data/training_test/")
    print("   # Then modify data/training to data/training_test in config")

    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()
