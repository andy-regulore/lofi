"""Pydantic schemas for configuration validation."""

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class QualityFilters(BaseModel):
    """Schema for quality filtering parameters."""

    min_tempo: float = Field(ge=0, le=300, description="Minimum tempo in BPM")
    max_tempo: float = Field(ge=0, le=300, description="Maximum tempo in BPM")
    min_duration: float = Field(ge=0, description="Minimum duration in seconds")
    max_duration: float = Field(ge=0, description="Maximum duration in seconds")
    require_drums: bool = Field(description="Whether drums are required")
    min_note_density: float = Field(ge=0, description="Minimum notes per second")
    max_note_density: float = Field(ge=0, description="Maximum notes per second")

    @model_validator(mode='after')
    def validate_ranges(self) -> 'QualityFilters':
        """Validate that min values are less than max values."""
        if self.min_tempo >= self.max_tempo:
            raise ValueError("min_tempo must be less than max_tempo")
        if self.min_duration >= self.max_duration:
            raise ValueError("min_duration must be less than max_duration")
        if self.min_note_density >= self.max_note_density:
            raise ValueError("min_note_density must be less than max_note_density")
        return self


class DataConfig(BaseModel):
    """Schema for data configuration."""

    midi_dir: str
    tokens_dir: str
    dataset_dir: str
    output_dir: str
    quality_filters: QualityFilters


class TokenizationConfig(BaseModel):
    """Schema for tokenization configuration."""

    tokenizer_type: Literal["REMI"] = "REMI"
    max_sequence_length: int = Field(gt=0, le=8192)
    chunk_size: int = Field(gt=0, le=8192)
    overlap: int = Field(ge=0)
    tempo_bins: int = Field(gt=0, le=128)
    velocity_bins: int = Field(gt=0, le=128)
    duration_bins: int = Field(gt=0, le=128)

    @model_validator(mode='after')
    def validate_overlap(self) -> 'TokenizationConfig':
        """Validate that overlap is less than chunk_size."""
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        return self


class ModelConfig(BaseModel):
    """Schema for model architecture configuration."""

    name: str
    vocab_size: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    num_layers: int = Field(gt=0, le=48)
    num_heads: int = Field(gt=0, le=32)
    context_length: int = Field(gt=0, le=8192)
    dropout: float = Field(ge=0.0, le=1.0)
    attention_dropout: float = Field(ge=0.0, le=1.0)

    @field_validator('num_heads')
    @classmethod
    def validate_num_heads(cls, v: int, info) -> int:
        """Validate that embedding_dim is divisible by num_heads."""
        # Note: embedding_dim might not be set yet, validate in model_validator
        return v

    @model_validator(mode='after')
    def validate_architecture(self) -> 'ModelConfig':
        """Validate model architecture constraints."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        return self


class TrainingConfig(BaseModel):
    """Schema for training configuration."""

    output_dir: str
    num_epochs: int = Field(gt=0, le=1000)
    batch_size: int = Field(gt=0, le=256)
    gradient_accumulation_steps: int = Field(gt=0, le=128)
    effective_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=1.0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0.0, le=1.0)
    max_grad_norm: float = Field(gt=0.0)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    fp16: bool = False
    logging_steps: int = Field(gt=0)
    eval_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)
    early_stopping_patience: Optional[int] = Field(None, gt=0)
    early_stopping_threshold: Optional[float] = Field(None, ge=0.0)
    target_eval_loss: Optional[float] = Field(None, gt=0.0)
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    dataloader_num_workers: int = Field(ge=0, le=32)

    @model_validator(mode='after')
    def validate_effective_batch_size(self) -> 'TrainingConfig':
        """Validate effective batch size calculation."""
        expected = self.batch_size * self.gradient_accumulation_steps
        if self.effective_batch_size != expected:
            raise ValueError(
                f"effective_batch_size ({self.effective_batch_size}) must equal "
                f"batch_size * gradient_accumulation_steps ({expected})"
            )
        return self


class ConditioningConfig(BaseModel):
    """Schema for generation conditioning configuration."""

    tempo_range: List[float] = Field(min_length=2, max_length=2)
    keys: List[str] = Field(min_length=1)
    moods: List[str] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_tempo_range(self) -> 'ConditioningConfig':
        """Validate tempo range."""
        if self.tempo_range[0] >= self.tempo_range[1]:
            raise ValueError("tempo_range[0] must be less than tempo_range[1]")
        if self.tempo_range[0] < 0 or self.tempo_range[1] > 300:
            raise ValueError("tempo_range must be between 0 and 300 BPM")
        return self


class GenerationConfig(BaseModel):
    """Schema for generation configuration."""

    num_tracks: int = Field(gt=0, le=10000)
    temperature: float = Field(gt=0.0, le=2.0)
    top_k: int = Field(gt=0, le=200)
    top_p: float = Field(gt=0.0, le=1.0)
    max_length: int = Field(gt=0, le=8192)
    conditioning: ConditioningConfig
    min_duration: float = Field(ge=0)
    max_duration: float = Field(ge=0)
    target_duration: float = Field(ge=0)

    @model_validator(mode='after')
    def validate_durations(self) -> 'GenerationConfig':
        """Validate duration constraints."""
        if self.min_duration >= self.max_duration:
            raise ValueError("min_duration must be less than max_duration")
        if not (self.min_duration <= self.target_duration <= self.max_duration):
            raise ValueError("target_duration must be between min_duration and max_duration")
        return self


class VinylCrackleConfig(BaseModel):
    """Schema for vinyl crackle effect configuration."""

    enabled: bool = True
    intensity: float = Field(ge=0.0, le=1.0)


class TapeWowFlutterConfig(BaseModel):
    """Schema for tape wow/flutter effect configuration."""

    enabled: bool = True
    depth: float = Field(ge=0.0, le=0.1)


class CompressionConfig(BaseModel):
    """Schema for compression effect configuration."""

    threshold_db: float = Field(ge=-60.0, le=0.0)
    ratio: float = Field(ge=1.0, le=20.0)
    attack_ms: float = Field(gt=0.0, le=100.0)
    release_ms: float = Field(gt=0.0, le=1000.0)


class LoFiEffectsConfig(BaseModel):
    """Schema for lo-fi effects configuration."""

    downsample_rate: int = Field(gt=8000, le=48000)
    lowpass_cutoff: float = Field(gt=100.0, le=22050.0)
    highpass_cutoff: float = Field(gt=0.0, le=1000.0)
    filter_order: int = Field(gt=0, le=10)
    bit_depth: int = Field(gt=4, le=24)
    vinyl_crackle: VinylCrackleConfig
    tape_wow_flutter: TapeWowFlutterConfig
    compression: CompressionConfig

    @model_validator(mode='after')
    def validate_filters(self) -> 'LoFiEffectsConfig':
        """Validate filter configuration."""
        if self.highpass_cutoff >= self.lowpass_cutoff:
            raise ValueError("highpass_cutoff must be less than lowpass_cutoff")
        return self


class AudioConfig(BaseModel):
    """Schema for audio processing configuration."""

    sample_rate: int = Field(gt=8000, le=96000)
    soundfont_path: Optional[str] = None
    lofi_effects: LoFiEffectsConfig
    target_lufs: float = Field(ge=-60.0, le=0.0)
    true_peak_max: float = Field(ge=-10.0, le=0.0)


class BatchGenerationConfig(BaseModel):
    """Schema for batch generation configuration."""

    num_tracks: int = Field(gt=0, le=10000)
    parallel_generation: int = Field(gt=0, le=32)
    quality_check: Dict[str, bool | float] = Field(
        default_factory=lambda: {"enabled": True, "min_score": 7.0}
    )
    ensure_tempo_variety: bool = True
    ensure_key_variety: bool = True
    ensure_mood_variety: bool = True
    create_metadata: bool = True
    generate_midi: bool = True
    generate_wav: bool = True
    generate_lofi: bool = True


class LoggingConfig(BaseModel):
    """Schema for logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    tensorboard: bool = True
    wandb: bool = False


class LoFiConfig(BaseModel):
    """Main configuration schema for the lo-fi music generator."""

    data: DataConfig
    tokenization: TokenizationConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    audio: AudioConfig
    batch_generation: BatchGenerationConfig
    instruments: Dict[str, List[int]] = Field(default_factory=dict)
    logging: LoggingConfig
    seed: int = Field(ge=0, le=2**32 - 1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'LoFiConfig':
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated LoFiConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration file
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False
            )

    def validate_paths(self) -> List[str]:
        """Validate that all directory paths exist or can be created.

        Returns:
            List of warning messages for missing paths
        """
        warnings = []
        paths_to_check = [
            self.data.midi_dir,
            self.data.tokens_dir,
            self.data.dataset_dir,
            self.data.output_dir,
            self.training.output_dir,
        ]

        for path_str in paths_to_check:
            path = Path(path_str)
            if not path.exists():
                warnings.append(f"Path does not exist: {path}")

        if self.audio.soundfont_path:
            sf_path = Path(self.audio.soundfont_path)
            if not sf_path.exists():
                warnings.append(f"Soundfont not found: {sf_path}")

        return warnings
