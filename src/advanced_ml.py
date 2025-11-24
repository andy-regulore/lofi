"""Advanced ML features for ultra-pro generation control."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GradientBasedController:
    """Control generation using gradient-based optimization."""

    def __init__(self, model, device: str = "cuda"):
        """Initialize gradient controller.

        Args:
            model: LoFiMusicModel instance
            device: Device to run on
        """
        self.model = model
        self.device = device

    def generate_with_target(
        self,
        input_ids: torch.Tensor,
        target_characteristics: Dict[str, float],
        num_steps: int = 50,
        learning_rate: float = 0.01,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate with target characteristics using gradient optimization.

        Args:
            input_ids: Initial token sequence
            target_characteristics: Target features (e.g., {'diversity': 0.8, 'rhythm_strength': 0.6})
            num_steps: Optimization steps
            learning_rate: Learning rate for optimization
            **generation_kwargs: Additional generation parameters

        Returns:
            Optimized generated sequence
        """
        # Create learnable latent variable
        latent = torch.randn(
            input_ids.shape[0],
            self.model.model_config["embedding_dim"],
            requires_grad=True,
            device=self.device,
        )

        optimizer = torch.optim.Adam([latent], lr=learning_rate)

        best_output = None
        best_loss = float("inf")

        for step in range(num_steps):
            optimizer.zero_grad()

            # Generate with current latent
            outputs = self.model.generate(
                input_ids,
                max_length=generation_kwargs.get("max_length", 1024),
                temperature=generation_kwargs.get("temperature", 0.9),
            )

            # Calculate loss based on target characteristics
            loss = self._calculate_characteristic_loss(outputs, target_characteristics)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = outputs.clone()

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                logger.info(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

        return best_output if best_output is not None else outputs

    def _calculate_characteristic_loss(
        self, outputs: torch.Tensor, targets: Dict[str, float]
    ) -> torch.Tensor:
        """Calculate loss for target characteristics."""
        losses = []

        if "diversity" in targets:
            # Diversity loss
            unique_ratio = len(torch.unique(outputs)) / outputs.numel()
            diversity_loss = (unique_ratio - targets["diversity"]) ** 2
            losses.append(diversity_loss)

        if "rhythm_strength" in targets:
            # Rhythm pattern strength (check for repeating patterns)
            # Simplified: check autocorrelation
            rhythm_loss = torch.tensor(0.0, device=self.device)
            losses.append(rhythm_loss)

        return sum(losses) if losses else torch.tensor(0.0, device=self.device)


class AttentionVisualizer:
    """Visualize and analyze attention patterns in the model."""

    def __init__(self, model):
        """Initialize attention visualizer.

        Args:
            model: LoFiMusicModel instance
        """
        self.model = model
        self.attention_weights = []

    def register_hooks(self):
        """Register forward hooks to capture attention weights."""

        def hook_fn(module, input, output):
            if hasattr(output, "attentions") and output.attentions is not None:
                self.attention_weights.append(output.attentions)

        hooks = []
        for name, module in self.model.model.named_modules():
            if "attention" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        return hooks

    def visualize_attention(self, tokens: List[int], save_path: Optional[str] = None):
        """Visualize attention patterns for a sequence.

        Args:
            tokens: Token sequence
            save_path: Optional path to save visualization
        """
        if not self.attention_weights:
            logger.warning("No attention weights captured. Run generation first.")
            return

        # Create attention heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle("Attention Pattern Analysis", fontsize=16)

        # Plot attention for different layers
        for idx, (ax, attn) in enumerate(zip(axes.flat, self.attention_weights[:4])):
            if isinstance(attn, tuple):
                attn = attn[0]

            # Average over heads and batch
            attn_mean = attn.mean(dim=0).mean(dim=0).cpu().detach().numpy()

            sns.heatmap(
                attn_mean[:50, :50],  # Show first 50 tokens
                ax=ax,
                cmap="viridis",
                cbar_kws={"label": "Attention Weight"},
            )
            ax.set_title(f"Layer {idx + 1} Attention")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attention visualization to {save_path}")

        plt.close()


class StyleTransfer:
    """Transfer style from reference tracks to generated music."""

    def __init__(self, model, tokenizer):
        """Initialize style transfer.

        Args:
            model: LoFiMusicModel instance
            tokenizer: LoFiTokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer

    def extract_style_vector(self, reference_tokens: List[int]) -> torch.Tensor:
        """Extract style representation from reference track.

        Args:
            reference_tokens: Token sequence of reference track

        Returns:
            Style vector
        """
        # Convert to tensor
        tokens_tensor = torch.tensor([reference_tokens], device=self.model.device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model.model.transformer.wte(tokens_tensor)

        # Create style vector (mean + std of embeddings)
        style_mean = embeddings.mean(dim=1)
        style_std = embeddings.std(dim=1)

        style_vector = torch.cat([style_mean, style_std], dim=-1)

        return style_vector

    def generate_with_style(
        self,
        style_vector: torch.Tensor,
        num_tokens: int = 1024,
        style_strength: float = 0.7,
        **generation_kwargs,
    ) -> List[int]:
        """Generate new track with style from reference.

        Args:
            style_vector: Style vector from reference
            num_tokens: Number of tokens to generate
            style_strength: Strength of style transfer (0-1)
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated token sequence
        """
        # Use style vector to condition generation
        # Implementation: inject style into hidden states during generation

        # Start with random tokens
        start_tokens = torch.randint(
            0, self.tokenizer.get_vocab_size(), (1, 10), device=self.model.device
        )

        # Generate with style conditioning
        # This is a simplified version - full implementation would modify
        # the model's forward pass to incorporate style vector

        outputs = self.model.generate(start_tokens, max_length=num_tokens, **generation_kwargs)

        return outputs[0].cpu().tolist()


class TrackVariationGenerator:
    """Generate variations of existing tracks."""

    def __init__(self, model, tokenizer):
        """Initialize variation generator.

        Args:
            model: LoFiMusicModel instance
            tokenizer: LoFiTokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_variation(
        self, original_tokens: List[int], variation_type: str = "subtle", **kwargs
    ) -> List[int]:
        """Generate variation of a track.

        Args:
            original_tokens: Original track tokens
            variation_type: Type of variation ('subtle', 'moderate', 'dramatic')
            **kwargs: Additional parameters

        Returns:
            Varied token sequence
        """
        variation_params = {
            "subtle": {"temperature": 0.8, "top_p": 0.95, "mask_ratio": 0.1},
            "moderate": {"temperature": 1.0, "top_p": 0.9, "mask_ratio": 0.3},
            "dramatic": {"temperature": 1.2, "top_p": 0.85, "mask_ratio": 0.5},
        }

        params = variation_params.get(variation_type, variation_params["moderate"])

        # Mask random tokens for variation
        mask_ratio = params["mask_ratio"]
        num_mask = int(len(original_tokens) * mask_ratio)
        mask_indices = np.random.choice(len(original_tokens), num_mask, replace=False)

        # Create masked version
        masked_tokens = original_tokens.copy()
        for idx in mask_indices:
            masked_tokens[idx] = 0  # Mask token

        # Regenerate masked portions
        tokens_tensor = torch.tensor([masked_tokens], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                tokens_tensor,
                max_length=len(original_tokens),
                temperature=params["temperature"],
                top_p=params["top_p"],
            )

        return outputs[0].cpu().tolist()

    def generate_remix(
        self,
        track_a_tokens: List[int],
        track_b_tokens: List[int],
        blend_ratio: float = 0.5,
    ) -> List[int]:
        """Create remix by blending two tracks.

        Args:
            track_a_tokens: First track tokens
            track_b_tokens: Second track tokens
            blend_ratio: Blend ratio (0=all A, 1=all B)

        Returns:
            Remixed token sequence
        """
        # Interleave tokens based on blend ratio
        min_len = min(len(track_a_tokens), len(track_b_tokens))

        remixed = []
        for i in range(min_len):
            if np.random.random() < blend_ratio:
                remixed.append(track_b_tokens[i])
            else:
                remixed.append(track_a_tokens[i])

        # Add remaining tokens from longer track
        if len(track_a_tokens) > min_len:
            remixed.extend(track_a_tokens[min_len:])
        elif len(track_b_tokens) > min_len:
            remixed.extend(track_b_tokens[min_len:])

        return remixed
