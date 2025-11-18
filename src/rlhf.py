"""Reinforcement Learning from Human Feedback (RLHF) for music generation.

This module implements RLHF to fine-tune the music generator based on human preferences.

Components:
- Reward model training from human comparisons
- PPO (Proximal Policy Optimization) for policy training
- DPO (Direct Preference Optimization) as alternative
- Preference data collection and management
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class MusicPreferenceDataset(Dataset):
    """Dataset for human music preferences (A vs B comparisons)."""

    def __init__(self, preferences_file: str):
        """Initialize preference dataset.

        Args:
            preferences_file: Path to JSON file with preferences

        Format: [
            {
                "track_a": [token_ids],
                "track_b": [token_ids],
                "preferred": "a" or "b",
                "metadata_a": {...},
                "metadata_b": {...},
                "reason": "optional explanation"
            }
        ]
        """
        with open(preferences_file) as f:
            self.preferences = json.load(f)

        logger.info(f"Loaded {len(self.preferences)} preference pairs")

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        pref = self.preferences[idx]

        return {
            'track_a': torch.tensor(pref['track_a'], dtype=torch.long),
            'track_b': torch.tensor(pref['track_b'], dtype=torch.long),
            'preferred': 0 if pref['preferred'] == 'a' else 1,
            'metadata_a': pref['metadata_a'],
            'metadata_b': pref['metadata_b'],
        }


class RewardModel(nn.Module):
    """Reward model that predicts human preference scores."""

    def __init__(self, base_model: nn.Module, hidden_dim: int = 768):
        """Initialize reward model.

        Args:
            base_model: Pre-trained music generation model
            hidden_dim: Hidden dimension size
        """
        super().__init__()

        # Use the base model's transformer
        self.transformer = base_model.model.transformer

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Freeze transformer initially (optional)
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to get reward score.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Reward scores [batch_size]
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids)
        hidden_states = outputs.last_hidden_state

        # Pool hidden states (mean pooling)
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]

        # Get reward
        reward = self.reward_head(pooled).squeeze(-1)  # [batch_size]

        return reward

    def unfreeze_transformer(self):
        """Unfreeze transformer for fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        logger.info("Transformer unfrozen for reward model training")


class RewardModelTrainer:
    """Trainer for reward model using preference data."""

    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        device: str = 'cuda',
    ):
        """Initialize reward trainer.

        Args:
            reward_model: Reward model to train
            learning_rate: Learning rate
            device: Device to train on
        """
        self.reward_model = reward_model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    def train_step(
        self,
        track_a: torch.Tensor,
        track_b: torch.Tensor,
        preferred: torch.Tensor,
    ) -> float:
        """Single training step.

        Args:
            track_a: First track tokens [batch_size, seq_len]
            track_b: Second track tokens [batch_size, seq_len]
            preferred: Which was preferred (0=a, 1=b) [batch_size]

        Returns:
            Loss value
        """
        self.reward_model.train()
        self.optimizer.zero_grad()

        # Get rewards
        reward_a = self.reward_model(track_a.to(self.device))
        reward_b = self.reward_model(track_b.to(self.device))

        # Bradley-Terry loss
        # P(a > b) = sigmoid(reward_a - reward_b)
        logits = reward_a - reward_b
        preferred = preferred.to(self.device).float()

        # Convert preferred to -1/1 format
        # If b preferred: -1, if a preferred: 1
        labels = 1 - 2 * preferred

        # Loss: -log P(preferred track has higher reward)
        loss = F.binary_cross_entropy_with_logits(logits, (labels + 1) / 2)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Preference data loader

        Returns:
            Training metrics
        """
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        for batch in dataloader:
            loss = self.train_step(
                batch['track_a'],
                batch['track_b'],
                batch['preferred'],
            )

            total_loss += loss
            num_batches += 1

            # Calculate accuracy
            with torch.no_grad():
                reward_a = self.reward_model(batch['track_a'].to(self.device))
                reward_b = self.reward_model(batch['track_b'].to(self.device))
                predicted = (reward_b > reward_a).long()
                correct += (predicted == batch['preferred'].to(self.device)).sum().item()
                total += len(predicted)

        return {
            'loss': total_loss / num_batches,
            'accuracy': correct / total,
        }

    def save(self, path: str):
        """Save reward model.

        Args:
            path: Save path
        """
        torch.save(self.reward_model.state_dict(), path)
        logger.info(f"Reward model saved to {path}")

    def load(self, path: str):
        """Load reward model.

        Args:
            path: Load path
        """
        self.reward_model.load_state_dict(torch.load(path))
        logger.info(f"Reward model loaded from {path}")


class PPOTrainer:
    """PPO (Proximal Policy Optimization) trainer for music generation."""

    def __init__(
        self,
        policy_model,
        reward_model: RewardModel,
        value_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = 'cuda',
    ):
        """Initialize PPO trainer.

        Args:
            policy_model: Policy model (music generator)
            reward_model: Trained reward model
            value_model: Value model (can be same as reward model)
            learning_rate: Learning rate
            clip_epsilon: PPO clipping parameter
            gamma: Discount factor
            lam: GAE lambda
            device: Device to train on
        """
        self.policy = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.value_model = value_model or reward_model
        self.device = device

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam

        self.optimizer = torch.optim.AdamW(
            self.policy.get_model().parameters(),
            lr=learning_rate,
        )

        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def generate_and_score(
        self,
        prompts: List[torch.Tensor],
        max_length: int = 1024,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Generate music and get rewards.

        Args:
            prompts: List of prompt tensors
            max_length: Maximum generation length

        Returns:
            Tuple of (generated_sequences, rewards)
        """
        self.policy.eval()

        generated = []
        rewards = []

        with torch.no_grad():
            for prompt in prompts:
                # Generate
                output = self.policy.generate(
                    prompt.unsqueeze(0).to(self.device),
                    max_length=max_length,
                )

                # Get reward
                reward = self.reward_model(output)

                generated.append(output.squeeze(0))
                rewards.append(reward.item())

        return generated, rewards

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of state values

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

            next_value = values[i]

        return advantages, returns

    def ppo_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Single PPO update step.

        Args:
            states: State sequences
            actions: Actions taken
            old_log_probs: Log probs from old policy
            advantages: Advantage estimates
            returns: Return estimates

        Returns:
            Training metrics
        """
        self.policy.train()

        # Get current policy outputs
        outputs = self.policy.get_model()(states)
        logits = outputs.logits

        # Compute log probs of actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        action_log_probs = action_log_probs.mean(dim=-1)  # Average over sequence

        # Compute ratio
        ratio = torch.exp(action_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_model(states)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.get_model().parameters(), 1.0)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item(),
        }


class DPOTrainer:
    """Direct Preference Optimization (DPO) - simpler alternative to PPO.

    DPO directly optimizes the policy using preference data without a separate reward model.
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        device: str = 'cuda',
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: Policy model to train
            reference_model: Reference model (frozen copy of initial policy)
            beta: KL penalty coefficient
            learning_rate: Learning rate
            device: Device to train on
        """
        self.policy = policy_model.to(device)
        self.reference = reference_model.to(device)
        self.beta = beta
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.policy.get_model().parameters(),
            lr=learning_rate,
        )

        # Freeze reference model
        for param in self.reference.get_model().parameters():
            param.requires_grad = False

    def get_log_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of sequence.

        Args:
            model: Model to use
            input_ids: Input token IDs

        Returns:
            Log probability of sequence
        """
        outputs = model.get_model()(input_ids)
        logits = outputs.logits

        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum(dim=-1)

    def dpo_loss(
        self,
        preferred: torch.Tensor,
        rejected: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss.

        Args:
            preferred: Preferred track tokens
            rejected: Rejected track tokens

        Returns:
            Tuple of (loss, metrics)
        """
        # Get log probs from policy and reference
        policy_preferred_logp = self.get_log_probs(self.policy, preferred)
        policy_rejected_logp = self.get_log_probs(self.policy, rejected)

        with torch.no_grad():
            ref_preferred_logp = self.get_log_probs(self.reference, preferred)
            ref_rejected_logp = self.get_log_probs(self.reference, rejected)

        # DPO loss
        policy_ratio = policy_preferred_logp - policy_rejected_logp
        ref_ratio = ref_preferred_logp - ref_rejected_logp

        logits = self.beta * (policy_ratio - ref_ratio)
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        accuracy = (logits > 0).float().mean().item()
        reward_diff = policy_ratio.mean().item()

        return loss, {
            'loss': loss.item(),
            'accuracy': accuracy,
            'reward_diff': reward_diff,
        }

    def train_step(
        self,
        track_a: torch.Tensor,
        track_b: torch.Tensor,
        preferred: torch.Tensor,
    ) -> Dict[str, float]:
        """Single DPO training step.

        Args:
            track_a: First track tokens
            track_b: Second track tokens
            preferred: Which was preferred (0=a, 1=b)

        Returns:
            Training metrics
        """
        self.policy.train()
        self.optimizer.zero_grad()

        # Separate preferred and rejected
        batch_size = track_a.shape[0]
        preferred_tracks = []
        rejected_tracks = []

        for i in range(batch_size):
            if preferred[i] == 0:
                preferred_tracks.append(track_a[i])
                rejected_tracks.append(track_b[i])
            else:
                preferred_tracks.append(track_b[i])
                rejected_tracks.append(track_a[i])

        preferred_batch = torch.stack(preferred_tracks).to(self.device)
        rejected_batch = torch.stack(rejected_tracks).to(self.device)

        # Compute loss
        loss, metrics = self.dpo_loss(preferred_batch, rejected_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.get_model().parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Preference data loader

        Returns:
            Training metrics
        """
        total_metrics = {'loss': 0.0, 'accuracy': 0.0, 'reward_diff': 0.0}
        num_batches = 0

        for batch in dataloader:
            metrics = self.train_step(
                batch['track_a'],
                batch['track_b'],
                batch['preferred'],
            )

            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics


class PreferenceCollector:
    """Tool for collecting human preference data."""

    def __init__(self, output_file: str):
        """Initialize preference collector.

        Args:
            output_file: Path to save preferences
        """
        self.output_file = output_file
        self.preferences = []

        # Load existing if available
        if Path(output_file).exists():
            with open(output_file) as f:
                self.preferences = json.load(f)

    def add_preference(
        self,
        track_a: List[int],
        track_b: List[int],
        preferred: str,
        metadata_a: Dict,
        metadata_b: Dict,
        reason: Optional[str] = None,
    ):
        """Add a preference pair.

        Args:
            track_a: First track tokens
            track_b: Second track tokens
            preferred: 'a' or 'b'
            metadata_a: Metadata for track A
            metadata_b: Metadata for track B
            reason: Optional explanation
        """
        self.preferences.append({
            'track_a': track_a,
            'track_b': track_b,
            'preferred': preferred,
            'metadata_a': metadata_a,
            'metadata_b': metadata_b,
            'reason': reason,
        })

        # Save immediately
        self.save()

    def save(self):
        """Save preferences to file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.preferences, f, indent=2)

    def get_statistics(self) -> Dict:
        """Get statistics about collected preferences.

        Returns:
            Statistics dictionary
        """
        if not self.preferences:
            return {'total': 0}

        preferred_a = sum(1 for p in self.preferences if p['preferred'] == 'a')
        preferred_b = sum(1 for p in self.preferences if p['preferred'] == 'b')

        return {
            'total': len(self.preferences),
            'preferred_a': preferred_a,
            'preferred_b': preferred_b,
            'balance': abs(preferred_a - preferred_b) / len(self.preferences),
        }
