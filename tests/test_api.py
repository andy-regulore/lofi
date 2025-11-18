"""Tests for FastAPI server."""

import pytest
from fastapi.testclient import TestClient
import yaml


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'model': {
            'embedding_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'context_length': 512,
            'dropout': 0.1,
            'attention_dropout': 0.1,
        },
        'tokenization': {
            'velocity_bins': 32,
            'tempo_bins': 32,
            'chunk_size': 512,
            'overlap': 128,
        },
        'generation': {
            'temperature': 0.9,
            'top_k': 50,
            'top_p': 0.95,
            'max_length': 512,
            'conditioning': {
                'tempo_range': [60, 95],
                'keys': ['C', 'Am', 'F', 'G'],
                'moods': ['chill', 'melancholic', 'upbeat'],
            },
        },
        'data': {
            'quality_filters': {
                'min_tempo': 60,
                'max_tempo': 95,
                'min_duration': 30,
                'max_duration': 300,
                'require_drums': False,
                'min_note_density': 0.5,
                'max_note_density': 10.0,
            },
        },
        'audio': {
            'sample_rate': 44100,
            'target_lufs': -14.0,
            'true_peak_max': -1.0,
            'lofi_effects': {
                'lowpass_cutoff': 3500,
                'highpass_cutoff': 80,
                'bit_depth': 12,
                'downsample_rate': 22050,
                'compression': {
                    'threshold_db': -20,
                    'ratio': 3.0,
                },
                'vinyl_crackle': {
                    'enabled': True,
                    'intensity': 0.015,
                },
                'tape_wow_flutter': {
                    'enabled': True,
                    'depth': 0.002,
                },
            },
        },
        'seed': 42,
        'api': {
            'output_dir': 'api_output',
        },
    }


@pytest.mark.skipif(True, reason="API tests require full model initialization")
def test_api_root(test_config):
    """Test API root endpoint."""
    from src.api import create_app

    app = create_app(test_config)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'online'
    assert 'endpoints' in data


@pytest.mark.skipif(True, reason="API tests require full model initialization")
def test_health_endpoint(test_config):
    """Test health check endpoint."""
    from src.api import create_app

    app = create_app(test_config)
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] in ['healthy', 'degraded']


@pytest.mark.skipif(True, reason="API tests require full model initialization")
def test_generate_endpoint(test_config):
    """Test generation endpoint."""
    from src.api import create_app

    app = create_app(test_config)
    client = TestClient(app)

    request_data = {
        'tempo': 75,
        'key': 'Am',
        'mood': 'chill',
        'max_length': 256,
        'temperature': 0.9,
        'return_midi': True,
        'return_audio': False,
    }

    response = client.post("/api/v1/generate", json=request_data)
    assert response.status_code in [200, 503]  # 503 if resources unavailable


def test_generation_request_validation():
    """Test request model validation."""
    from src.api import GenerationRequest

    # Valid request
    request = GenerationRequest(
        tempo=75,
        key='Am',
        mood='chill',
        temperature=0.9,
    )
    assert request.tempo == 75

    # Invalid tempo
    with pytest.raises(ValueError):
        GenerationRequest(tempo=300)  # Out of range

    # Invalid key
    with pytest.raises(ValueError):
        GenerationRequest(key='invalid')

    # Invalid mood
    with pytest.raises(ValueError):
        GenerationRequest(mood='invalid')


def test_metrics_endpoint(test_config):
    """Test Prometheus metrics endpoint."""
    from src.api import create_app

    app = create_app(test_config)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
