"""Performance optimization utilities for production deployment.

Features:
- Model quantization (int8, float16)
- KV-cache optimization
- ONNX export for inference
- Generation caching
- Batch inference optimization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import numpy as np

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize models for faster inference and reduced memory."""

    @staticmethod
    def quantize_int8(model: nn.Module) -> nn.Module:
        """Quantize model to INT8.

        Args:
            model: PyTorch model

        Returns:
            Quantized model
        """
        logger.info("Quantizing model to INT8...")

        # Dynamic quantization (easiest for transformers)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )

        # Measure size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024

        logger.info(f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({quantized_size/original_size*100:.1f}%)")

        return quantized_model

    @staticmethod
    def convert_to_fp16(model: nn.Module, device: str = 'cuda') -> nn.Module:
        """Convert model to FP16 for faster GPU inference.

        Args:
            model: PyTorch model
            device: Device to move model to

        Returns:
            FP16 model
        """
        if not torch.cuda.is_available() or device == 'cpu':
            logger.warning("FP16 requires CUDA, skipping conversion")
            return model

        logger.info("Converting model to FP16...")

        model = model.to(device)
        model = model.half()

        logger.info("Model converted to FP16")

        return model

    @staticmethod
    def save_quantized(model: nn.Module, save_path: str):
        """Save quantized model.

        Args:
            model: Quantized model
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path / "quantized_model.pth")
        logger.info(f"Quantized model saved to {save_path}")


class GenerationCache:
    """Cache generated sequences for faster repeated generation."""

    def __init__(self, max_cache_size: int = 1000):
        """Initialize cache.

        Args:
            max_cache_size: Maximum number of cached generations
        """
        self.cache = {}
        self.access_count = {}
        self.max_size = max_cache_size

    def _get_cache_key(self, params: Dict) -> str:
        """Generate cache key from parameters.

        Args:
            params: Generation parameters

        Returns:
            Cache key
        """
        # Create deterministic hash of parameters
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def get(self, params: Dict) -> Optional[List[int]]:
        """Get cached generation.

        Args:
            params: Generation parameters

        Returns:
            Cached tokens or None
        """
        key = self._get_cache_key(params)

        if key in self.cache:
            self.access_count[key] += 1
            logger.debug(f"Cache hit for key {key[:8]}...")
            return self.cache[key]

        return None

    def put(self, params: Dict, tokens: List[int]):
        """Cache generation result.

        Args:
            params: Generation parameters
            tokens: Generated tokens
        """
        key = self._get_cache_key(params)

        # Evict least-used item if cache is full
        if len(self.cache) >= self.max_size:
            # Find least accessed key
            least_used = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used]
            del self.access_count[least_used]
            logger.debug(f"Evicted cache entry {least_used[:8]}...")

        self.cache[key] = tokens
        self.access_count[key] = 0
        logger.debug(f"Cached generation with key {key[:8]}...")

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size * 100,
            'total_accesses': sum(self.access_count.values()),
        }


class ONNXExporter:
    """Export models to ONNX format for optimized inference."""

    @staticmethod
    def export_model(
        model: GPT2LMHeadModel,
        output_path: str,
        vocab_size: int,
        seq_length: int = 1024,
        opset_version: int = 14,
    ):
        """Export model to ONNX format.

        Args:
            model: GPT-2 model
            output_path: Path to save ONNX model
            vocab_size: Vocabulary size
            seq_length: Sequence length
            opset_version: ONNX opset version
        """
        logger.info("Exporting model to ONNX...")

        model.eval()

        # Create dummy input
        dummy_input = torch.randint(0, vocab_size, (1, seq_length))

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'},
            }
        )

        logger.info(f"Model exported to ONNX: {output_path}")

        # Verify export
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")
        except ImportError:
            logger.warning("ONNX package not installed, skipping verification")


class BeamSearchGenerator:
    """Advanced generation with beam search."""

    @staticmethod
    def beam_search(
        model: GPT2LMHeadModel,
        input_ids: torch.Tensor,
        num_beams: int = 5,
        max_length: int = 1024,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
        device: str = 'cuda',
    ) -> Tuple[torch.Tensor, float]:
        """Generate with beam search.

        Args:
            model: GPT-2 model
            input_ids: Input token IDs
            num_beams: Number of beams
            max_length: Maximum generation length
            temperature: Sampling temperature
            length_penalty: Length penalty (>1 = longer, <1 = shorter)
            early_stopping: Stop when all beams finish
            no_repeat_ngram_size: Prevent repeating n-grams
            device: Device to run on

        Returns:
            Tuple of (generated_ids, score)
        """
        model.eval()
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]

        # Calculate score
        if hasattr(outputs, 'sequences_scores'):
            score = outputs.sequences_scores[0].item()
        else:
            score = 0.0

        return generated_ids, score


class ConstrainedDecoder:
    """Constrained decoding with music theory rules."""

    def __init__(self, music_theory_engine, tokenizer):
        """Initialize constrained decoder.

        Args:
            music_theory_engine: MusicTheoryEngine instance
            tokenizer: LoFiTokenizer instance
        """
        self.theory = music_theory_engine
        self.tokenizer = tokenizer
        self.allowed_tokens = None

    def set_constraints(self, key: str = 'C', allow_chromatic: bool = True):
        """Set music theory constraints.

        Args:
            key: Musical key
            allow_chromatic: Allow chromatic notes
        """
        # Get allowed pitch classes from key
        if key in self.theory.key_signatures:
            scale_notes = set(self.theory.key_signatures[key]['scale'])
            self.allowed_tokens = scale_notes
        else:
            self.allowed_tokens = None

        logger.info(f"Set constraints for key {key}, chromatic={allow_chromatic}")

    def constrained_logits_processor(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Process logits to enforce constraints.

        Args:
            input_ids: Input token IDs
            logits: Model logits

        Returns:
            Constrained logits
        """
        if self.allowed_tokens is None:
            return logits

        # Create mask for allowed tokens
        mask = torch.ones_like(logits) * float('-inf')

        # Allow tokens that match constraints
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0

        # Apply mask
        constrained_logits = logits + mask

        return constrained_logits

    def generate_constrained(
        self,
        model: GPT2LMHeadModel,
        input_ids: torch.Tensor,
        key: str = 'C',
        max_length: int = 1024,
        **kwargs
    ) -> torch.Tensor:
        """Generate with music theory constraints.

        Args:
            model: GPT-2 model
            input_ids: Input token IDs
            key: Musical key for constraints
            max_length: Maximum length
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        self.set_constraints(key=key)

        # Use logits processor
        from transformers import LogitsProcessorList

        logits_processor = LogitsProcessorList([
            lambda input_ids, logits: self.constrained_logits_processor(input_ids, logits)
        ])

        outputs = model.generate(
            input_ids,
            max_length=max_length,
            logits_processor=logits_processor,
            **kwargs
        )

        return outputs


class BatchInferenceOptimizer:
    """Optimize batch inference for high throughput."""

    @staticmethod
    def dynamic_batching(
        requests: List[Dict],
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
    ) -> List[List[Dict]]:
        """Create dynamic batches from requests.

        Args:
            requests: List of generation requests
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time for batching

        Returns:
            List of batches
        """
        batches = []
        current_batch = []

        for request in requests:
            current_batch.append(request)

            if len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []

        # Add remaining requests
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Created {len(batches)} batches from {len(requests)} requests")

        return batches

    @staticmethod
    def batch_generate(
        model: GPT2LMHeadModel,
        input_ids_batch: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate for batch of inputs.

        Args:
            model: GPT-2 model
            input_ids_batch: List of input tensors
            **kwargs: Generation parameters

        Returns:
            List of generated sequences
        """
        # Pad to same length
        max_len = max(ids.shape[1] for ids in input_ids_batch)

        padded_batch = []
        attention_masks = []

        for ids in input_ids_batch:
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                padded = torch.cat([ids, torch.zeros(ids.shape[0], pad_len, dtype=torch.long)], dim=1)
                mask = torch.cat([torch.ones(ids.shape[1]), torch.zeros(pad_len)])
            else:
                padded = ids
                mask = torch.ones(ids.shape[1])

            padded_batch.append(padded)
            attention_masks.append(mask)

        # Stack into batch
        batch_input_ids = torch.cat(padded_batch, dim=0)
        batch_attention_mask = torch.stack(attention_masks)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                **kwargs
            )

        # Split back into individual sequences
        results = [outputs[i:i+1] for i in range(outputs.shape[0])]

        return results


class ModelPruner:
    """Prune models to reduce size and increase speed."""

    @staticmethod
    def magnitude_pruning(
        model: nn.Module,
        amount: float = 0.3,
    ) -> nn.Module:
        """Prune model weights by magnitude.

        Args:
            model: PyTorch model
            amount: Fraction of weights to prune (0-1)

        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune

        logger.info(f"Pruning {amount*100:.1f}% of model weights...")

        # Prune linear layers
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        logger.info("Pruning complete")

        return model


class KVCacheOptimizer:
    """Optimize KV-cache for faster autoregressive generation."""

    @staticmethod
    def enable_kv_cache(model: GPT2LMHeadModel):
        """Enable KV-cache in model config.

        Args:
            model: GPT-2 model
        """
        model.config.use_cache = True
        logger.info("KV-cache enabled")

    @staticmethod
    def clear_kv_cache(model: GPT2LMHeadModel):
        """Clear KV-cache to free memory.

        Args:
            model: GPT-2 model
        """
        if hasattr(model, '_past_key_values'):
            model._past_key_values = None
        logger.debug("KV-cache cleared")


# Convenience function for applying all optimizations
def optimize_model_for_production(
    model: GPT2LMHeadModel,
    quantization: str = 'fp16',  # 'none', 'fp16', 'int8'
    pruning_amount: float = 0.0,
    device: str = 'cuda',
) -> GPT2LMHeadModel:
    """Apply all production optimizations to model.

    Args:
        model: GPT-2 model
        quantization: Quantization method
        pruning_amount: Amount of pruning (0-1)
        device: Target device

    Returns:
        Optimized model
    """
    logger.info("Optimizing model for production...")

    # Pruning
    if pruning_amount > 0:
        model = ModelPruner.magnitude_pruning(model, amount=pruning_amount)

    # Quantization
    if quantization == 'fp16':
        model = ModelQuantizer.convert_to_fp16(model, device=device)
    elif quantization == 'int8':
        model = ModelQuantizer.quantize_int8(model)

    # Enable KV-cache
    KVCacheOptimizer.enable_kv_cache(model)

    logger.info("Production optimization complete")

    return model
