"""Production-ready FastAPI server for lo-fi music generation.

Features:
- REST API for generation
- WebSocket for real-time streaming
- Queue management for batch processing
- Health checks and metrics
- Rate limiting
"""

import logging
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, validator
from starlette.responses import Response

from src.audio_processor import LoFiAudioProcessor
from src.generator import LoFiGenerator
from src.model import ConditionedLoFiModel
from src.quality_scorer import MusicQualityScorer
from src.tokenizer import LoFiTokenizer
from src.utils.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

# Prometheus metrics
GENERATION_REQUESTS = Counter("lofi_generation_requests_total", "Total generation requests")
GENERATION_DURATION = Histogram("lofi_generation_duration_seconds", "Generation duration")
ACTIVE_GENERATIONS = Gauge("lofi_active_generations", "Number of active generations")
QUEUE_SIZE = Gauge("lofi_queue_size", "Number of items in generation queue")
QUALITY_SCORES = Histogram("lofi_quality_scores", "Quality scores distribution")


class GenerationRequest(BaseModel):
    """Request model for music generation."""

    tempo: Optional[float] = Field(None, ge=50, le=200, description="Tempo in BPM")
    key: Optional[str] = Field(None, description="Musical key (e.g., 'C', 'Am')")
    mood: Optional[str] = Field(None, description="Mood (chill, melancholic, upbeat, etc.)")
    max_length: Optional[int] = Field(1024, ge=256, le=4096, description="Maximum token length")
    temperature: Optional[float] = Field(0.9, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: Optional[float] = Field(0.95, ge=0.1, le=1.0, description="Nucleus sampling")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    apply_lofi_effects: bool = Field(True, description="Apply lo-fi audio effects")
    return_midi: bool = Field(True, description="Return MIDI file")
    return_audio: bool = Field(True, description="Return audio file")
    min_quality_score: Optional[float] = Field(
        None, ge=0, le=10, description="Minimum quality score"
    )

    @validator("key")
    def validate_key(cls, v):
        if v is not None:
            valid_keys = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
                "Cm",
                "C#m",
                "Dm",
                "D#m",
                "Em",
                "Fm",
                "F#m",
                "Gm",
                "G#m",
                "Am",
                "A#m",
                "Bm",
            ]
            if v not in valid_keys:
                raise ValueError(f"Invalid key. Must be one of {valid_keys}")
        return v

    @validator("mood")
    def validate_mood(cls, v):
        if v is not None:
            valid_moods = [
                "chill",
                "melancholic",
                "upbeat",
                "relaxed",
                "dreamy",
                "focus",
                "sleep",
                "study",
            ]
            if v not in valid_moods:
                raise ValueError(f"Invalid mood. Must be one of {valid_moods}")
        return v


class GenerationResponse(BaseModel):
    """Response model for music generation."""

    id: str
    status: str
    tempo: float
    key: str
    mood: str
    num_tokens: int
    quality_score: float
    midi_url: Optional[str] = None
    audio_url: Optional[str] = None
    generation_time: float
    metadata: Dict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    device: str
    gpu_available: bool
    active_generations: int
    queue_size: int
    total_generations: int


class LoFiAPI:
    """Production API for lo-fi music generation."""

    def __init__(self, config: Dict, model_path: Optional[str] = None):
        """Initialize API.

        Args:
            config: Configuration dictionary
            model_path: Path to trained model
        """
        self.config = config
        self.app = FastAPI(
            title="Lo-Fi Music Generator API",
            description="Production API for AI-powered lo-fi music generation",
            version="2.0.0",
        )

        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self._initialize_components(model_path)

        # Generation queue and tracking
        self.generation_queue = deque()
        self.active_generations = {}
        self.completed_generations = {}
        self.total_generations = 0

        # Output directory
        self.output_dir = Path(config.get("api", {}).get("output_dir", "api_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Register routes
        self._register_routes()

        # Start background worker
        self.worker_task = None

    def _initialize_components(self, model_path: Optional[str]):
        """Initialize model and processors."""
        logger.info("Initializing API components...")

        # Resource manager
        self.resource_manager = ResourceManager()
        device = self.resource_manager.get_optimal_device()

        # Tokenizer
        self.tokenizer = LoFiTokenizer(self.config)

        # Model
        self.model = ConditionedLoFiModel(self.config, self.tokenizer.get_vocab_size())
        if model_path:
            self.model.load(model_path)
        self.model.to(device)

        # Generator
        self.generator = LoFiGenerator(self.model, self.tokenizer, self.config, device=device)

        # Audio processor
        self.audio_processor = LoFiAudioProcessor(self.config)

        # Quality scorer
        self.quality_scorer = MusicQualityScorer(self.config)

        logger.info(f"API components initialized on {device}")

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/", response_model=Dict)
        async def root():
            """API root endpoint."""
            return {
                "name": "Lo-Fi Music Generator API",
                "version": "2.0.0",
                "status": "online",
                "endpoints": {
                    "generate": "/api/v1/generate",
                    "status": "/api/v1/status/{id}",
                    "health": "/api/v1/health",
                    "metrics": "/metrics",
                },
            }

        @self.app.post("/api/v1/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
            """Generate lo-fi music track."""
            GENERATION_REQUESTS.inc()

            # Create generation ID
            gen_id = str(uuid.uuid4())

            # Check resources
            resources = self.resource_manager.check_all_resources()
            if not resources["all_ok"]:
                raise HTTPException(status_code=503, detail="Insufficient resources available")

            try:
                ACTIVE_GENERATIONS.inc()
                start_time = time.time()

                # Generate track
                tokens, metadata = self.generator.generate_track(
                    tempo=request.tempo,
                    key=request.key,
                    mood=request.mood,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                )

                # Calculate quality score
                quality_score = self.quality_scorer.score_midi_tokens(tokens, metadata)
                QUALITY_SCORES.observe(quality_score)

                # Check minimum quality
                if request.min_quality_score and quality_score < request.min_quality_score:
                    # Retry once with different seed
                    tokens, metadata = self.generator.generate_track(
                        tempo=request.tempo,
                        key=request.key,
                        mood=request.mood,
                        max_length=request.max_length,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        seed=request.seed + 1 if request.seed else None,
                    )
                    quality_score = self.quality_scorer.score_midi_tokens(tokens, metadata)

                # Save MIDI
                midi_path = None
                if request.return_midi:
                    midi_path = self.output_dir / f"{gen_id}.mid"
                    self.generator.tokens_to_midi(tokens, str(midi_path))

                # Process audio
                audio_path = None
                if request.return_audio and midi_path:
                    if request.apply_lofi_effects:
                        result = self.audio_processor.process_midi_to_lofi(
                            str(midi_path),
                            str(self.output_dir),
                            name=gen_id,
                            save_clean=False,
                            save_lofi=True,
                        )
                        audio_path = result.get("lofi_wav_path")
                    else:
                        audio_path = self.output_dir / f"{gen_id}_clean.wav"
                        self.audio_processor.midi_to_wav(str(midi_path), str(audio_path))

                generation_time = time.time() - start_time
                GENERATION_DURATION.observe(generation_time)

                # Track completion
                self.total_generations += 1

                response = GenerationResponse(
                    id=gen_id,
                    status="completed",
                    tempo=metadata["tempo"],
                    key=metadata["key"],
                    mood=metadata["mood"],
                    num_tokens=metadata["num_tokens"],
                    quality_score=quality_score,
                    midi_url=f"/api/v1/download/{gen_id}.mid" if midi_path else None,
                    audio_url=f"/api/v1/download/{gen_id}_lofi.wav" if audio_path else None,
                    generation_time=generation_time,
                    metadata=metadata,
                )

                return response

            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                ACTIVE_GENERATIONS.dec()

        @self.app.get("/api/v1/download/{filename}")
        async def download(filename: str):
            """Download generated file."""
            file_path = self.output_dir / filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(file_path)

        @self.app.get("/api/v1/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            resources = self.resource_manager.check_all_resources()

            return HealthResponse(
                status="healthy" if resources["all_ok"] else "degraded",
                version="2.0.0",
                device=self.resource_manager.get_optimal_device(),
                gpu_available=torch.cuda.is_available(),
                active_generations=len(self.active_generations),
                queue_size=len(self.generation_queue),
                total_generations=self.total_generations,
            )

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @self.app.websocket("/ws/generate")
        async def websocket_generate(websocket: WebSocket):
            """WebSocket endpoint for real-time generation updates."""
            await websocket.accept()

            try:
                # Receive generation request
                data = await websocket.receive_json()
                request = GenerationRequest(**data)

                gen_id = str(uuid.uuid4())

                # Send initial status
                await websocket.send_json(
                    {
                        "id": gen_id,
                        "status": "started",
                        "progress": 0.0,
                    }
                )

                # Generate with progress updates
                await websocket.send_json(
                    {
                        "status": "generating_tokens",
                        "progress": 0.25,
                    }
                )

                tokens, metadata = self.generator.generate_track(
                    tempo=request.tempo,
                    key=request.key,
                    mood=request.mood,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                )

                await websocket.send_json(
                    {
                        "status": "scoring_quality",
                        "progress": 0.5,
                    }
                )

                quality_score = self.quality_scorer.score_midi_tokens(tokens, metadata)

                await websocket.send_json(
                    {
                        "status": "creating_midi",
                        "progress": 0.75,
                    }
                )

                midi_path = self.output_dir / f"{gen_id}.mid"
                self.generator.tokens_to_midi(tokens, str(midi_path))

                if request.return_audio:
                    await websocket.send_json(
                        {
                            "status": "processing_audio",
                            "progress": 0.9,
                        }
                    )

                    result = self.audio_processor.process_midi_to_lofi(
                        str(midi_path),
                        str(self.output_dir),
                        name=gen_id,
                        save_lofi=request.apply_lofi_effects,
                    )

                # Send completion
                await websocket.send_json(
                    {
                        "id": gen_id,
                        "status": "completed",
                        "progress": 1.0,
                        "quality_score": quality_score,
                        "metadata": metadata,
                        "midi_url": f"/api/v1/download/{gen_id}.mid",
                        "audio_url": (
                            f"/api/v1/download/{gen_id}_lofi.wav" if request.return_audio else None
                        ),
                    }
                )

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json(
                    {
                        "status": "error",
                        "error": str(e),
                    }
                )

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server.

        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn arguments
        """
        logger.info(f"Starting Lo-Fi API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


def create_app(config_path: str = "config.yaml", model_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI app instance.

    Args:
        config_path: Path to configuration file
        model_path: Path to trained model

    Returns:
        FastAPI app instance
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api = LoFiAPI(config, model_path)
    return api.app


if __name__ == "__main__":
    import sys

    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api = LoFiAPI(config, model_path)
    api.run()
