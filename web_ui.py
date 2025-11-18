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

        # Step 1: Generate musical composition
        generation_status['current_step'] = 'Composing music...'
        generation_status['progress'] = 20

        sample_rate = 44100
        tempo_bpm = 75
        beat_duration = 60.0 / tempo_bpm
        bar_duration = beat_duration * 4

        # Key/mood-specific chord progressions
        key_root = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
                    'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
                    'Cm': 0, 'Dm': 2, 'Em': 4, 'Fm': 5, 'Gm': 7, 'Am': 9, 'Bm': 11}.get(key, 9)
        base_freq = 440 * (2 ** ((key_root - 9) / 12))  # A4 = 440 Hz

        # Chord progressions (semitone intervals from root)
        progressions = {
            'chill': [[0, 4, 7], [-5, -1, 2], [-3, 0, 4], [-5, -1, 2]],  # I-IV-vi-IV
            'melancholic': [[0, 3, 7], [-5, -2, 2], [-8, -5, -1], [-3, 0, 4]],  # i-iv-bVI-VI
            'upbeat': [[0, 4, 7], [-3, 0, 4], [-5, -1, 2], [-7, -3, 0]],  # I-vi-IV-V
            'dreamy': [[0, 4, 7, 11], [-3, 0, 4, 7], [-5, -1, 2, 7], [2, 7, 11]],  # With 7ths
            'relaxed': [[0, 4, 7], [2, 5, 9], [-5, -1, 2], [0, 4, 7]]  # I-ii-IV-I
        }
        progression = progressions.get(mood, progressions['chill'])

        # Generate audio
        audio = np.zeros(int(sample_rate * duration))
        t = np.arange(len(audio)) / sample_rate

        # Step 2: Layer the composition
        generation_status['current_step'] = 'Adding chords, bass, and melody...'
        generation_status['progress'] = 35

        num_bars = int(duration / bar_duration)

        for bar_idx in range(num_bars):
            chord_idx = bar_idx % len(progression)
            chord_intervals = progression[chord_idx]
            bar_start = bar_idx * bar_duration
            bar_end = (bar_idx + 1) * bar_duration

            start_sample = int(bar_start * sample_rate)
            end_sample = int(bar_end * sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)

            bar_t = t[start_sample:end_sample]

            # Add rich chords (piano-like)
            for interval in chord_intervals:
                freq = base_freq * (2 ** (interval / 12)) * (2 ** 1)  # One octave up
                # Multiple harmonics for richer sound
                audio[start_sample:end_sample] += 0.15 * np.sin(2 * np.pi * freq * bar_t)
                audio[start_sample:end_sample] += 0.08 * np.sin(2 * np.pi * freq * 2 * bar_t)  # 2nd harmonic
                audio[start_sample:end_sample] += 0.04 * np.sin(2 * np.pi * freq * 3 * bar_t)  # 3rd harmonic

            # Add bass line (root notes, octave down)
            bass_freq = base_freq * (2 ** (chord_intervals[0] / 12)) / 2
            # Bass on beats 1 and 3
            for beat in [0, 2]:
                beat_start = bar_start + beat * beat_duration
                beat_end = beat_start + beat_duration * 0.8
                beat_start_sample = int(beat_start * sample_rate)
                beat_end_sample = int(beat_end * sample_rate)
                if beat_end_sample > len(audio):
                    beat_end_sample = len(audio)

                beat_t = t[beat_start_sample:beat_end_sample]
                envelope = np.exp(-3 * (beat_t - beat_start))
                audio[beat_start_sample:beat_end_sample] += 0.25 * np.sin(2 * np.pi * bass_freq * beat_t) * envelope

            # Add melody (pentatonic scale)
            if bar_idx % 2 == 1 or mood == 'upbeat':  # Every other bar
                pentatonic = [0, 2, 4, 7, 9]  # Pentatonic scale degrees
                for _ in range(np.random.randint(3, 7)):
                    note_start = bar_start + np.random.uniform(0, bar_duration - beat_duration * 0.5)
                    note_dur = beat_duration * np.random.choice([0.25, 0.5, 0.75])
                    note_end = min(note_start + note_dur, bar_end)

                    note_start_sample = int(note_start * sample_rate)
                    note_end_sample = int(note_end * sample_rate)
                    if note_end_sample > len(audio):
                        break

                    degree = np.random.choice(pentatonic)
                    melody_freq = base_freq * (2 ** (degree / 12)) * (2 ** 2)  # Two octaves up
                    note_t = t[note_start_sample:note_end_sample]
                    envelope = np.exp(-4 * (note_t - note_start))
                    audio[note_start_sample:note_end_sample] += 0.12 * np.sin(2 * np.pi * melody_freq * note_t) * envelope

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Step 3: Add ambient
        generation_status['current_step'] = f'Adding {theme} ambience...'
        generation_status['progress'] = 50

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

        # Step 4: Apply LoFi effects
        generation_status['current_step'] = 'Applying LoFi effects...'
        generation_status['progress'] = 70

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
    response = make_response(render_template('index_comprehensive.html'))
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

# Training endpoints
training_status = {
    'is_training': False,
    'epoch': 0,
    'total_epochs': 0,
    'loss': 0.0,
    'status': '',
    'error': None,
    'last_trained': None
}

@app.route('/api/upload/training', methods=['POST'])
def upload_training():
    """Upload training MIDI or WAV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '' or not (file.filename.endswith('.mid') or file.filename.endswith('.midi') or file.filename.endswith('.wav')):
        return jsonify({'error': 'Invalid file. MIDI or WAV files only (.mid, .midi, .wav)'}), 400

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{filename}"

    # Save to training data folder
    training_data_dir = Path('data/training')
    training_data_dir.mkdir(parents=True, exist_ok=True)

    filepath = training_data_dir / unique_filename
    file.save(str(filepath))

    return jsonify({
        'status': 'success',
        'filename': unique_filename,
        'original_filename': filename,
        'path': str(filepath)
    })

@app.route('/api/training/files')
def list_training_files():
    """List all training files"""
    training_data_dir = Path('data/training')
    training_data_dir.mkdir(parents=True, exist_ok=True)

    files = []
    # Get both MIDI and WAV files
    all_files = list(training_data_dir.glob('*.mid*')) + list(training_data_dir.glob('*.wav'))
    for file in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
        files.append({
            'filename': file.name,
            'original_name': '_'.join(file.name.split('_')[1:]),
            'size': file.stat().st_size,
            'uploaded': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })

    return jsonify(files)

@app.route('/api/training/files/<filename>', methods=['DELETE'])
def delete_training_file(filename):
    """Delete training file"""
    filepath = Path('data/training') / secure_filename(filename)

    if filepath.exists():
        filepath.unlink()
        return jsonify({'status': 'deleted'})
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    global training_status

    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400

    config = request.json
    epochs = int(config.get('epochs', 10))
    batch_size = int(config.get('batch_size', 8))
    learning_rate = float(config.get('learning_rate', 0.0001))

    # Check if we have training data
    training_data_dir = Path('data/training')
    training_files = list(training_data_dir.glob('*.mid*')) + list(training_data_dir.glob('*.wav'))

    if len(training_files) == 0:
        return jsonify({'error': 'No training data uploaded. Please upload MIDI or WAV files first.'}), 400

    # Start training in background
    def training_task():
        global training_status

        try:
            training_status['is_training'] = True
            training_status['total_epochs'] = epochs
            training_status['status'] = 'Initializing...'
            training_status['error'] = None

            # Simulate training for now (TODO: integrate actual training)
            for epoch in range(1, epochs + 1):
                training_status['epoch'] = epoch
                training_status['status'] = f'Training epoch {epoch}/{epochs}'

                # Simulate epoch time
                time.sleep(2)

                # Simulate decreasing loss
                training_status['loss'] = 5.0 * (1 - epoch / epochs) + 0.1

            training_status['is_training'] = False
            training_status['status'] = 'Training completed'
            training_status['last_trained'] = datetime.now().isoformat()

        except Exception as e:
            training_status['error'] = str(e)
            training_status['is_training'] = False

    thread = threading.Thread(target=training_task)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/training/status')
def get_training_status():
    """Get training status"""
    return jsonify(training_status)

# Batch generation endpoints
batch_generation_status = {
    'is_generating': False,
    'total': 0,
    'completed': 0,
    'progress': 0,
    'status': '',
    'error': None
}

@app.route('/api/batch/generate', methods=['POST'])
def batch_generate():
    """Start batch generation"""
    global batch_generation_status

    if batch_generation_status['is_generating']:
        return jsonify({'error': 'Batch generation already in progress'}), 400

    config = request.json
    count = int(config.get('count', 5))
    mood = config.get('mood', 'random')
    duration = int(config.get('duration', 180))

    def batch_task():
        global batch_generation_status

        try:
            batch_generation_status['is_generating'] = True
            batch_generation_status['total'] = count
            batch_generation_status['completed'] = 0
            batch_generation_status['error'] = None

            moods = ['chill', 'melancholic', 'upbeat', 'dreamy', 'relaxed']
            keys = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']

            for i in range(count):
                batch_generation_status['status'] = f'Generating track {i+1}/{count}'

                # Select mood
                selected_mood = mood if mood != 'random' else np.random.choice(moods)
                selected_key = np.random.choice(keys)

                # Generate using the same logic as single generation
                settings = {
                    'mood': selected_mood,
                    'theme': 'plain',
                    'lofi_preset': 'medium',
                    'duration': duration,
                    'key': selected_key
                }

                # Call the generation task
                generate_audio_task(settings)

                batch_generation_status['completed'] = i + 1
                batch_generation_status['progress'] = int(((i + 1) / count) * 100)

                # Small delay between generations
                time.sleep(1)

            batch_generation_status['is_generating'] = False
            batch_generation_status['status'] = 'Batch generation completed'

        except Exception as e:
            batch_generation_status['error'] = str(e)
            batch_generation_status['is_generating'] = False

    thread = threading.Thread(target=batch_task)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/batch/status')
def get_batch_status():
    """Get batch generation status"""
    return jsonify(batch_generation_status)

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 LoFi Music Empire - Web UI")
    print("=" * 60)
    print("\n✅ Server starting...")
    print("\n🌐 Open your browser to: http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
