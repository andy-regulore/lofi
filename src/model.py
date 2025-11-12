"""GPT-2 based transformer model for lo-fi music generation.

This module implements a 117M parameter GPT-2 model configured for MIDI token generation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoFiMusicModel:
    """GPT-2 based model for lo-fi music generation."""

    def __init__(self, config: Dict, vocab_size: int):
        """Initialize the model.

        Args:
            config: Configuration dictionary
            vocab_size: Size of the token vocabulary
        """
        self.config = config
        self.model_config = config['model']
        self.vocab_size = vocab_size

        # Update vocab size in model config
        self.model_config['vocab_size'] = vocab_size

        # Create GPT-2 configuration (117M parameters)
        self.gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=self.model_config['context_length'],
            n_embd=self.model_config['embedding_dim'],
            n_layer=self.model_config['num_layers'],
            n_head=self.model_config['num_heads'],
            resid_pdrop=self.model_config['dropout'],
            embd_pdrop=self.model_config['dropout'],
            attn_pdrop=self.model_config['attention_dropout'],
            use_cache=True,
        )

        # Initialize model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the GPT-2 model."""
        logger.info("Initializing GPT-2 model with config:")
        logger.info(f"  Vocabulary size: {self.gpt2_config.vocab_size}")
        logger.info(f"  Embedding dim: {self.gpt2_config.n_embd}")
        logger.info(f"  Layers: {self.gpt2_config.n_layer}")
        logger.info(f"  Heads: {self.gpt2_config.n_head}")
        logger.info(f"  Context length: {self.gpt2_config.n_positions}")

        self.model = GPT2LMHeadModel(self.gpt2_config)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    def get_model(self) -> GPT2LMHeadModel:
        """Get the underlying PyTorch model.

        Returns:
            GPT2LMHeadModel instance
        """
        return self.model

    def save(self, save_path: str):
        """Save model to disk.

        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path)

        # Save config
        import json
        config_path = save_path / 'lofi_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load(self, load_path: str):
        """Load model from disk.

        Args:
            load_path: Directory to load model from
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise ValueError(f"Model path does not exist: {load_path}")

        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(load_path)

        logger.info(f"Model loaded from {load_path}")

    def to(self, device: str):
        """Move model to device.

        Args:
            device: Device to move to ('cuda' or 'cpu')
        """
        self.model = self.model.to(device)
        logger.info(f"Model moved to {device}")
        return self

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate token sequences.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_return_sequences: Number of sequences to generate
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs [batch_size * num_return_sequences, seq_len]
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                do_sample=True,
                **kwargs
            )

        return outputs

    def get_model_info(self) -> Dict:
        """Get model information.

        Returns:
            Dictionary with model statistics
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': num_params,
            'trainable_parameters': num_trainable,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.gpt2_config.n_embd,
            'num_layers': self.gpt2_config.n_layer,
            'num_heads': self.gpt2_config.n_head,
            'context_length': self.gpt2_config.n_positions,
        }


class ConditionedLoFiModel(LoFiMusicModel):
    """Extended model with conditioning tokens for tempo, key, and mood.

    This model prepends special conditioning tokens to the input sequence
    to guide generation towards specific musical characteristics.
    """

    def __init__(self, config: Dict, vocab_size: int):
        """Initialize conditioned model.

        Args:
            config: Configuration dictionary
            vocab_size: Base vocabulary size (conditioning tokens added on top)
        """
        # Reserve tokens for conditioning
        self.num_conditioning_tokens = 100  # tempo (32) + key (12) + mood (5) + extras

        # Adjust vocab size to include conditioning tokens
        super().__init__(config, vocab_size + self.num_conditioning_tokens)

        self.base_vocab_size = vocab_size
        self.conditioning_offset = vocab_size

        # Define conditioning token ranges
        self.tempo_token_start = self.conditioning_offset
        self.tempo_token_count = 32
        self.key_token_start = self.tempo_token_start + self.tempo_token_count
        self.key_token_count = 12
        self.mood_token_start = self.key_token_start + self.key_token_count
        self.mood_token_count = 10

        logger.info(f"Conditioned model vocab size: {self.vocab_size} "
                   f"(base: {self.base_vocab_size}, conditioning: {self.num_conditioning_tokens})")

    def get_tempo_token(self, tempo: float) -> int:
        """Get conditioning token for a given tempo.

        Args:
            tempo: Tempo in BPM

        Returns:
            Token ID for tempo
        """
        # Map tempo to token index (50-200 BPM range)
        tempo_idx = int((tempo - 50) / 150 * self.tempo_token_count)
        tempo_idx = max(0, min(self.tempo_token_count - 1, tempo_idx))
        return self.tempo_token_start + tempo_idx

    def get_key_token(self, key: str) -> int:
        """Get conditioning token for a given key.

        Args:
            key: Musical key (e.g., 'C', 'Am', 'F#')

        Returns:
            Token ID for key
        """
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        }
        # Extract root note (handle minor keys like 'Am')
        root = key.replace('m', '').replace('maj', '')
        key_idx = key_map.get(root, 0)
        return self.key_token_start + key_idx

    def get_mood_token(self, mood: str) -> int:
        """Get conditioning token for a given mood.

        Args:
            mood: Mood descriptor

        Returns:
            Token ID for mood
        """
        mood_map = {
            'chill': 0,
            'melancholic': 1,
            'upbeat': 2,
            'relaxed': 3,
            'dreamy': 4,
            'focus': 5,
            'sleep': 6,
            'study': 7,
        }
        mood_idx = mood_map.get(mood.lower(), 0)
        return self.mood_token_start + mood_idx

    def create_conditioning_prefix(self, tempo: float = 75, key: str = 'C',
                                   mood: str = 'chill') -> list:
        """Create conditioning token prefix.

        Args:
            tempo: Desired tempo in BPM
            key: Desired musical key
            mood: Desired mood

        Returns:
            List of conditioning token IDs
        """
        return [
            self.get_tempo_token(tempo),
            self.get_key_token(key),
            self.get_mood_token(mood),
        ]
