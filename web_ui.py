"""
LoFi Music Empire - Web UI
Complete interface for all features
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, make_response
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import threading
import os
from werkzeug.utils import secure_filename

# Import our modules
from src.metadata_generator import MetadataGenerator
from src.copyright_protection import CopyrightDatabase, CopyrightProtector
from src.lofi_effects import LoFiEffectsChain
from src.ambient_sounds import AmbientSoundGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mid', 'midi', 'wav', 'mp3', 'flac', 'ogg'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Global state
generation_status = {
    'is_generating': False,
    'progress': 0,
    'current_step': '',
    'track_id': None,
    'error': None
}

# Initialize modules
metadata_gen = MetadataGenerator()
copyright_db = CopyrightDatabase('data/copyright.db')
copyright_protector = CopyrightProtector(copyright_db)
lofi_effects = LoFiEffectsChain()
ambient_gen = AmbientSoundGenerator()

# Ensure directories exist
output_dir = Path('output')
(output_dir / 'audio').mkdir(parents=True, exist_ok=True)
(output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

def generate_audio_task(settings):
    """Background task for audio generation"""
    global generation_status

    try:
        generation_status['is_generating'] = True
        generation_status['progress'] = 0
        generation_status['error'] = None

        # Extract settings
        mood = settings.get('mood', 'chill')
        theme = settings.get('theme', 'plain')
        lofi_preset = settings.get('lofi_preset', 'medium')
        duration = int(settings.get('duration', 180))
        key = settings.get('key', 'Am')

        track_id = f"track_{int(time.time())}"
        audio_path = output_dir / 'audio' / f"{track_id}.wav"
        generation_status['track_id'] = track_id

        # Step 1: Generate base music
        generation_status['current_step'] = 'Generating base music...'
        generation_status['progress'] = 20

        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Chord progressions
        chord_map = {
            'chill': [261.63, 196.00, 220.00, 174.61],
            'melancholic': [220.00, 174.61, 196.00, 146.83],
            'upbeat': [261.63, 293.66, 196.00, 261.63],
            'relaxed': [174.61, 261.63, 196.00, 220.00],
            'dreamy': [220.00, 196.00, 174.61, 261.63]
        }

        freqs = chord_map.get(mood, chord_map['chill'])
        audio = np.zeros(len(t))
        beats_per_bar = 4
        beat_duration = duration / (beats_per_bar * 4)

        for i, freq in enumerate(freqs):
            start_idx = int(i * beat_duration * 4 * sample_rate)
            end_idx = int((i + 1) * beat_duration * 4 * sample_rate)
            if end_idx > len(t):
                end_idx = len(t)

            segment = t[start_idx:end_idx]
            audio[start_idx:end_idx] += 0.20 * np.sin(2 * np.pi * freq * segment)
            audio[start_idx:end_idx] += 0.15 * np.sin(2 * np.pi * freq * 1.5 * segment)
            audio[start_idx:end_idx] += 0.12 * np.sin(2 * np.pi * freq * 1.25 * segment)
            audio[start_idx:end_idx] += 0.08 * np.sin(2 * np.pi * freq * 0.5 * segment)

            melody_freq = freq * 2
            melody = 0.10 * np.sin(2 * np.pi * melody_freq * segment + np.sin(segment * 0.5))
            audio[start_idx:end_idx] += melody

        audio = audio / np.max(np.abs(audio))

        # Step 2: Add ambient
        generation_status['current_step'] = f'Adding {theme} ambience...'
        generation_status['progress'] = 40

        if theme == 'rain':
            ambient = ambient_gen.generate_rain(duration, intensity='medium', include_thunder=False)
            mix_level = 0.15
        elif theme == 'cafe':
            ambient = ambient_gen.generate_cafe_ambience(duration, busyness='medium')
            mix_level = 0.12
        elif theme == 'urban_chill':
            ambient = np.random.randn(len(audio)) * 0.03
            for _ in range(5):
                pos = np.random.randint(0, len(audio) - sample_rate * 3)
                car_sound = np.sin(2 * np.pi * 80 * np.arange(sample_rate * 3) / sample_rate)
                car_sound *= np.exp(-np.arange(sample_rate * 3) / sample_rate)
                if pos + len(car_sound) <= len(ambient):
                    ambient[pos:pos+len(car_sound)] += car_sound * 0.05
            mix_level = 0.10
        elif theme == 'nature':
            ambient = ambient_gen.generate_nature_sounds(duration, environment='forest')
            mix_level = 0.15
        else:
            ambient = np.zeros(len(audio))
            mix_level = 0.0

        if mix_level > 0:
            if len(ambient) > len(audio):
                ambient = ambient[:len(audio)]
            elif len(ambient) < len(audio):
                ambient = np.pad(ambient, (0, len(audio) - len(ambient)))
            audio = audio * (1 - mix_level) + ambient * mix_level

        # Step 3: Apply LoFi effects
        generation_status['current_step'] = 'Applying LoFi effects...'
        generation_status['progress'] = 60

        audio_lofi = lofi_effects.process_full_chain(audio, preset=lofi_preset)

        # Fade
        fade_samples = int(0.5 * sample_rate)
        audio_lofi[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio_lofi[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        audio_lofi = audio_lofi / np.max(np.abs(audio_lofi)) * 0.75

        # Save
        generation_status['current_step'] = 'Saving audio...'
        generation_status['progress'] = 80

        sf.write(str(audio_path), audio_lofi, sample_rate)

        # Generate metadata
        generation_status['current_step'] = 'Generating metadata...'
        generation_status['progress'] = 90

        metadata = metadata_gen.generate_complete_metadata(
            mood=mood,
            style='lofi',
            use_case='study',
            bpm=75,
            key=key,
            duration=duration
        )

        metadata_path = output_dir / 'metadata' / f"{track_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Complete
        generation_status['current_step'] = 'Complete!'
        generation_status['progress'] = 100
        generation_status['is_generating'] = False

    except Exception as e:
        generation_status['error'] = str(e)
        generation_status['is_generating'] = False
        generation_status['progress'] = 0

@app.route('/')
def index():
    """Main page"""
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/generate', methods=['POST'])
def generate():
    """Start audio generation"""
    if generation_status['is_generating']:
        return jsonify({'error': 'Generation already in progress'}), 400

    settings = request.json

    # Start generation in background thread
    thread = threading.Thread(target=generate_audio_task, args=(settings,))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/status')
def status():
    """Get generation status"""
    return jsonify(generation_status)

@app.route('/api/tracks')
def list_tracks():
    """List all generated tracks"""
    tracks = []
    audio_dir = output_dir / 'audio'
    metadata_dir = output_dir / 'metadata'

    for audio_file in sorted(audio_dir.glob('*.wav'), key=lambda x: x.stat().st_mtime, reverse=True):
        track_id = audio_file.stem
        metadata_file = metadata_dir / f"{track_id}.json"

        track_info = {
            'id': track_id,
            'filename': audio_file.name,
            'size': audio_file.stat().st_size,
            'created': datetime.fromtimestamp(audio_file.stat().st_mtime).isoformat(),
        }

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                track_info['title'] = metadata.get('title', 'Untitled')
                track_info['mood'] = metadata.get('mood', 'unknown')
                track_info['duration'] = metadata.get('duration_seconds', 0)

        tracks.append(track_info)

    return jsonify(tracks)

@app.route('/api/track/<track_id>')
def get_track(track_id):
    """Get track details"""
    metadata_file = output_dir / 'metadata' / f"{track_id}.json"

    if not metadata_file.exists():
        return jsonify({'error': 'Track not found'}), 404

    with open(metadata_file) as f:
        metadata = json.load(f)

    return jsonify(metadata)

@app.route('/api/download/<track_id>')
def download_track(track_id):
    """Download audio file"""
    audio_file = output_dir / 'audio' / f"{track_id}.wav"

    if not audio_file.exists():
        return jsonify({'error': 'Track not found'}), 404

    return send_file(audio_file, as_attachment=True)

@app.route('/api/delete/<track_id>', methods=['DELETE'])
def delete_track(track_id):
    """Delete a track"""
    audio_file = output_dir / 'audio' / f"{track_id}.wav"
    metadata_file = output_dir / 'metadata' / f"{track_id}.json"

    deleted = False
    if audio_file.exists():
        audio_file.unlink()
        deleted = True
    if metadata_file.exists():
        metadata_file.unlink()
        deleted = True

    if deleted:
        return jsonify({'status': 'deleted'})
    else:
        return jsonify({'error': 'Track not found'}), 404

@app.route('/api/upload/midi', methods=['POST'])
def upload_midi():
    """Upload MIDI file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        return jsonify({
            'status': 'success',
            'filename': unique_filename,
            'original_filename': filename,
            'path': filepath
        })
    else:
        return jsonify({'error': 'Invalid file type. Allowed: MIDI, WAV, MP3, FLAC, OGG'}), 400

@app.route('/api/uploads')
def list_uploads():
    """List all uploaded files"""
    uploads = []
    upload_path = Path(app.config['UPLOAD_FOLDER'])

    for file in sorted(upload_path.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True):
        if file.is_file():
            uploads.append({
                'filename': file.name,
                'original_name': '_'.join(file.name.split('_')[1:]),  # Remove timestamp
                'size': file.stat().st_size,
                'uploaded': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })

    return jsonify(uploads)

@app.route('/api/upload/<filename>/delete', methods=['DELETE'])
def delete_upload(filename):
    """Delete uploaded file"""
    filepath = Path(app.config['UPLOAD_FOLDER']) / secure_filename(filename)

    if filepath.exists():
        filepath.unlink()
        return jsonify({'status': 'deleted'})
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 LoFi Music Empire - Web UI")
    print("=" * 60)
    print("\n✅ Server starting...")
    print("\n🌐 Open your browser to: http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
