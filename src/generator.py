"""Music generation module for lo-fi tracks.

Handles:
- Loading trained models
- Conditional generation (tempo, key, mood)
- Token sampling strategies
- MIDI file generation
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoFiGenerator:
    """Generator for lo-fi music tracks."""

    def __init__(self, model, tokenizer, config: Dict, device: str = "cuda"):
        """Initialize generator.

        Args:
            model: Trained LoFiMusicModel instance
            tokenizer: LoFiTokenizer instance
            config: Configuration dictionary
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.gen_config = config["generation"]
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.get_model().eval()

        logger.info(f"Generator initialized on {device}")

    def generate_track(
        self,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        mood: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], Dict]:
        """Generate a single lo-fi track.

        Args:
            tempo: Target tempo in BPM (random if None)
            key: Target key (random if None)
            mood: Target mood (random if None)
            max_length: Maximum token sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility

        Returns:
            Tuple of (generated_tokens, metadata)
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Use defaults from config if not specified
        if tempo is None:
            tempo_range = self.gen_config["conditioning"]["tempo_range"]
            tempo = random.uniform(tempo_range[0], tempo_range[1])

        if key is None:
            key = random.choice(self.gen_config["conditioning"]["keys"])

        if mood is None:
            mood = random.choice(self.gen_config["conditioning"]["moods"])

        max_length = max_length or self.gen_config["max_length"]
        temperature = temperature or self.gen_config["temperature"]
        top_k = top_k or self.gen_config["top_k"]
        top_p = top_p or self.gen_config["top_p"]

        logger.info(f"Generating track: tempo={tempo:.1f}, key={key}, mood={mood}")

        # Create conditioning prefix if model supports it
        if hasattr(self.model, "create_conditioning_prefix"):
            conditioning_tokens = self.model.create_conditioning_prefix(tempo, key, mood)
            input_ids = torch.tensor([conditioning_tokens], dtype=torch.long).to(self.device)
        else:
            # Start with a random token from vocabulary
            start_token = random.randint(0, self.tokenizer.get_vocab_size() - 1)
            input_ids = torch.tensor([[start_token]], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=0,
            )

        # Convert to list
        generated_tokens = output_ids[0].cpu().tolist()

        # Metadata
        metadata = {
            "tempo": tempo,
            "key": key,
            "mood": mood,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_tokens": len(generated_tokens),
            "seed": seed,
        }

        logger.info(f"Generated {len(generated_tokens)} tokens")

        return generated_tokens, metadata

    def tokens_to_midi(self, tokens: List[int], output_path: str) -> bool:
        """Convert generated tokens to MIDI file.

        Args:
            tokens: List of token IDs
            output_path: Path to save MIDI file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove conditioning tokens if present
            if hasattr(self.model, "base_vocab_size"):
                # Filter out conditioning tokens
                base_vocab_size = self.model.base_vocab_size
                tokens = [t for t in tokens if t < base_vocab_size]

            # Convert to MIDI
            self.tokenizer.tokens_to_midi(tokens, output_path)
            logger.info(f"Saved MIDI to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error converting tokens to MIDI: {e}")
            return False

    def generate_and_save(
        self,
        output_path: str,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        mood: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Generate track and save as MIDI.

        Args:
            output_path: Path to save MIDI file
            tempo: Target tempo
            key: Target key
            mood: Target mood
            **kwargs: Additional generation parameters

        Returns:
            Metadata dictionary
        """
        # Generate tokens
        tokens, metadata = self.generate_track(tempo=tempo, key=key, mood=mood, **kwargs)

        # Save as MIDI
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = self.tokens_to_midi(tokens, str(output_path))

        if success:
            metadata["output_path"] = str(output_path)
        else:
            metadata["error"] = "Failed to convert to MIDI"

        return metadata

    def batch_generate(
        self,
        num_tracks: int,
        output_dir: str,
        name_prefix: str = "lofi_track",
        ensure_variety: bool = True,
    ) -> List[Dict]:
        """Generate multiple tracks.

        Args:
            num_tracks: Number of tracks to generate
            output_dir: Directory to save tracks
            name_prefix: Prefix for track filenames
            ensure_variety: Ensure variety in tempo, key, mood

        Returns:
            List of metadata dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_list = []

        # Prepare variety if requested
        if ensure_variety:
            tempos = np.linspace(
                self.gen_config["conditioning"]["tempo_range"][0],
                self.gen_config["conditioning"]["tempo_range"][1],
                num_tracks,
            )
            keys = self.gen_config["conditioning"]["keys"] * (
                num_tracks // len(self.gen_config["conditioning"]["keys"]) + 1
            )
            moods = self.gen_config["conditioning"]["moods"] * (
                num_tracks // len(self.gen_config["conditioning"]["moods"]) + 1
            )

            random.shuffle(keys)
            random.shuffle(moods)
        else:
            tempos = [None] * num_tracks
            keys = [None] * num_tracks
            moods = [None] * num_tracks

        logger.info(f"Generating {num_tracks} tracks...")

        for i in range(num_tracks):
            logger.info(f"Generating track {i+1}/{num_tracks}")

            output_path = output_dir / f"{name_prefix}_{i+1:03d}.mid"

            metadata = self.generate_and_save(
                output_path=str(output_path),
                tempo=float(tempos[i]) if tempos[i] is not None else None,
                key=keys[i],
                mood=moods[i],
                seed=self.config["seed"] + i,
            )

            metadata["track_number"] = i + 1
            metadata_list.append(metadata)

        logger.info(f"Generated {num_tracks} tracks in {output_dir}")

        # Save metadata
        import json

        metadata_file = output_dir / f"{name_prefix}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata_list, f, indent=2)

        return metadata_list

    def calculate_quality_score(self, tokens: List[int], metadata: Dict) -> float:
        """Calculate a quality score for generated music.

        This is a heuristic score based on:
        - Token sequence length (longer is better, up to a point)
        - Token diversity (more unique tokens is better)
        - Structural features (patterns, repetition)

        Args:
            tokens: Generated token sequence
            metadata: Generation metadata

        Returns:
            Quality score (0-10)
        """
        score = 5.0  # Base score

        # Length score (prefer 1000-2000 tokens)
        target_length = 1500
        length_diff = abs(len(tokens) - target_length)
        length_score = max(0, 3 - (length_diff / 500))
        score += length_score

        # Diversity score
        unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
        diversity_score = unique_ratio * 2  # 0-2 points
        score += diversity_score

        # Bonus for appropriate tempo
        if 65 <= metadata.get("tempo", 0) <= 85:
            score += 0.5

        # Cap at 10
        score = min(10.0, score)

        return score
