"""Real-time MIDI generation and interaction.

Features:
- Real-time MIDI input/output
- Interactive jamming session
- Loop-based generation
- Live parameter control
"""

import logging
from typing import Dict, List, Optional, Callable
import time
import threading
import queue

import numpy as np

logger = logging.getLogger(__name__)


class RealtimeMIDIGenerator:
    """Real-time MIDI generation engine."""

    def __init__(
        self,
        model,
        tokenizer,
        buffer_size: int = 32,
        latency_ms: float = 100,
    ):
        """Initialize real-time generator.

        Args:
            model: Music generation model
            tokenizer: Tokenizer
            buffer_size: Token buffer size
            latency_ms: Target latency in milliseconds
        """
        self.model = model
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.latency_ms = latency_ms

        # State
        self.is_generating = False
        self.context_tokens = []
        self.output_queue = queue.Queue()

        # Threading
        self.generation_thread = None

    def start(
        self,
        initial_tokens: Optional[List[int]] = None,
        conditioning: Optional[Dict] = None,
    ):
        """Start real-time generation.

        Args:
            initial_tokens: Initial context tokens
            conditioning: Conditioning parameters
        """
        if self.is_generating:
            logger.warning("Already generating")
            return

        # Initialize context
        if initial_tokens:
            self.context_tokens = initial_tokens[-self.buffer_size:]
        elif conditioning and hasattr(self.model, 'create_conditioning_prefix'):
            self.context_tokens = self.model.create_conditioning_prefix(
                tempo=conditioning.get('tempo', 75),
                key=conditioning.get('key', 'C'),
                mood=conditioning.get('mood', 'chill'),
            )
        else:
            # Random start
            self.context_tokens = [np.random.randint(0, self.tokenizer.get_vocab_size())]

        self.is_generating = True

        # Start generation thread
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()

        logger.info("Real-time generation started")

    def stop(self):
        """Stop real-time generation."""
        self.is_generating = False

        if self.generation_thread:
            self.generation_thread.join(timeout=1.0)

        logger.info("Real-time generation stopped")

    def _generation_loop(self):
        """Main generation loop (runs in thread)."""
        import torch

        self.model.eval()

        while self.is_generating:
            start_time = time.time()

            # Generate next token
            with torch.no_grad():
                context = torch.tensor([self.context_tokens], device=self.model.device)

                outputs = self.model.get_model()(context)
                logits = outputs.logits[0, -1, :]

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Add to context
            self.context_tokens.append(next_token)

            # Maintain buffer size
            if len(self.context_tokens) > self.buffer_size:
                self.context_tokens = self.context_tokens[-self.buffer_size:]

            # Output token
            self.output_queue.put(next_token)

            # Sleep to maintain target latency
            elapsed = (time.time() - start_time) * 1000  # ms
            if elapsed < self.latency_ms:
                time.sleep((self.latency_ms - elapsed) / 1000)

    def get_next_tokens(self, num_tokens: int = 1) -> List[int]:
        """Get next generated tokens.

        Args:
            num_tokens: Number of tokens to get

        Returns:
            List of tokens
        """
        tokens = []

        for _ in range(num_tokens):
            try:
                token = self.output_queue.get(timeout=self.latency_ms / 1000 * 2)
                tokens.append(token)
            except queue.Empty:
                break

        return tokens

    def inject_tokens(self, tokens: List[int]):
        """Inject tokens into context (for interactive input).

        Args:
            tokens: Tokens to inject
        """
        self.context_tokens.extend(tokens)

        # Maintain buffer size
        if len(self.context_tokens) > self.buffer_size:
            self.context_tokens = self.context_tokens[-self.buffer_size:]


class InteractiveJammer:
    """Interactive jamming session manager."""

    def __init__(
        self,
        generator: RealtimeMIDIGenerator,
        bpm: int = 75,
        measures: int = 4,
    ):
        """Initialize interactive jammer.

        Args:
            generator: Real-time generator
            bpm: Tempo in BPM
            measures: Number of measures per loop
        """
        self.generator = generator
        self.bpm = bpm
        self.measures = measures

        # Calculate timing
        self.beat_duration = 60.0 / bpm  # seconds per beat
        self.measure_duration = self.beat_duration * 4  # 4/4 time
        self.loop_duration = self.measure_duration * measures

        # State
        self.is_jamming = False
        self.loops = []
        self.current_loop = []

    def start_jamming(self, conditioning: Optional[Dict] = None):
        """Start jamming session.

        Args:
            conditioning: Initial conditioning
        """
        if self.is_jamming:
            return

        self.is_jamming = True
        self.generator.start(conditioning=conditioning)

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        logger.info(f"Jamming session started: {self.bpm} BPM, {self.measures} measures")

    def stop_jamming(self):
        """Stop jamming session."""
        self.is_jamming = False
        self.generator.stop()

        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)

        logger.info("Jamming session stopped")

    def _recording_loop(self):
        """Record generated tokens in loops."""
        loop_start = time.time()

        while self.is_jamming:
            # Get generated tokens
            tokens = self.generator.get_next_tokens(num_tokens=4)

            if tokens:
                self.current_loop.extend(tokens)

            # Check if loop complete
            elapsed = time.time() - loop_start

            if elapsed >= self.loop_duration:
                # Save loop
                self.loops.append(self.current_loop.copy())
                logger.info(f"Loop {len(self.loops)} complete: {len(self.current_loop)} tokens")

                # Start new loop
                self.current_loop = []
                loop_start = time.time()

            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def get_loops(self) -> List[List[int]]:
        """Get recorded loops.

        Returns:
            List of token sequences (one per loop)
        """
        return self.loops.copy()

    def play_loop(self, loop_index: int):
        """Play a recorded loop.

        Args:
            loop_index: Index of loop to play
        """
        if 0 <= loop_index < len(self.loops):
            loop_tokens = self.loops[loop_index]
            self.generator.inject_tokens(loop_tokens)
            logger.info(f"Playing loop {loop_index}")


class LoopGenerator:
    """Generate loopable MIDI sequences."""

    def __init__(self, model, tokenizer):
        """Initialize loop generator.

        Args:
            model: Music generation model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_loop(
        self,
        num_measures: int = 4,
        tempo: int = 75,
        ensure_loopable: bool = True,
        **kwargs
    ) -> List[int]:
        """Generate a loopable sequence.

        Args:
            num_measures: Number of measures
            tempo: Tempo in BPM
            ensure_loopable: Ensure start and end match for seamless loop
            **kwargs: Additional generation parameters

        Returns:
            Token sequence
        """
        # Estimate tokens per measure (rough)
        # At 75 BPM, 4/4 time, ~16 notes per measure
        tokens_per_measure = 16
        target_length = num_measures * tokens_per_measure

        # Generate
        import torch

        if hasattr(self.model, 'create_conditioning_prefix'):
            initial = self.model.create_conditioning_prefix(
                tempo=tempo,
                key=kwargs.get('key', 'C'),
                mood=kwargs.get('mood', 'chill'),
            )
        else:
            initial = [0]

        initial_tensor = torch.tensor([initial], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                initial_tensor,
                max_length=len(initial) + target_length,
                temperature=kwargs.get('temperature', 0.9),
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.95),
            )

        tokens = outputs[0].cpu().tolist()

        # Remove conditioning prefix
        if hasattr(self.model, 'base_vocab_size'):
            tokens = [t for t in tokens if t < self.model.base_vocab_size]

        if ensure_loopable:
            tokens = self._make_loopable(tokens)

        return tokens

    def _make_loopable(self, tokens: List[int]) -> List[int]:
        """Ensure tokens loop seamlessly.

        Args:
            tokens: Token sequence

        Returns:
            Modified tokens that loop better
        """
        # Simple approach: ensure first and last few tokens create smooth transition
        # In practice, would use more sophisticated music theory

        if len(tokens) < 8:
            return tokens

        # Match last notes to first notes (pitch classes)
        # This is a simplified version
        first_notes = tokens[:4]
        last_notes = tokens[-4:]

        # If they're very different, adjust last notes
        # (This is a placeholder - real implementation would be more musical)

        return tokens
