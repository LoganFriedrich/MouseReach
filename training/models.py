"""
Neural Network Model Architectures for MouseReach Deep Learning Pipeline.

Three model architectures:
1. BoundaryDetector: TCN or Transformer for binary classification at each frame
2. ReachDetector: BiLSTM-CRF for sequence labeling (reach start/end detection)
3. OutcomeClassifier: MLP on aggregated features for outcome classification

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import (
    Config, get_config,
    BoundaryDetectorConfig, ReachDetectorConfig, OutcomeClassifierConfig
)


# ==============================================================================
# UTILITY LAYERS
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalBlock(nn.Module):
    """Single block of Temporal Convolutional Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, seq_len)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out + residual)
        return out


class TCN(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(
        self,
        input_size: int,
        channels: List[int],
        kernel_size: int = 7,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]

            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, features)

        Returns:
            Tensor of shape (batch, seq_len, channels[-1])
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        return x


# ==============================================================================
# BOUNDARY DETECTOR
# ==============================================================================

class BoundaryDetectorTCN(nn.Module):
    """
    Boundary detector using Temporal Convolutional Network.

    Takes a window of features and predicts whether the center frame is a boundary.
    """

    def __init__(
        self,
        input_size: int,
        config: Optional[BoundaryDetectorConfig] = None
    ):
        super().__init__()
        self.config = config or get_config().boundary_detector

        self.tcn = TCN(
            input_size=input_size,
            channels=self.config.tcn_channels,
            kernel_size=self.config.tcn_kernel_size,
            dropout=self.config.tcn_dropout
        )

        # Global pooling + classification head
        hidden_size = self.config.tcn_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.tcn_dropout),
            nn.Linear(hidden_size // 2, self.config.n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, window_size, features)

        Returns:
            Tensor of shape (batch, n_classes) - logits
        """
        # TCN forward
        features = self.tcn(x)  # (batch, window_size, hidden)

        # Use center frame features
        center_idx = features.size(1) // 2
        center_features = features[:, center_idx, :]  # (batch, hidden)

        # Classification
        logits = self.classifier(center_features)
        return logits


class BoundaryDetectorTransformer(nn.Module):
    """
    Boundary detector using Transformer architecture.
    """

    def __init__(
        self,
        input_size: int,
        config: Optional[BoundaryDetectorConfig] = None
    ):
        super().__init__()
        self.config = config or get_config().boundary_detector

        d_model = self.config.transformer_d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout=self.config.transformer_dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.transformer_nhead,
            dim_feedforward=self.config.transformer_dim_feedforward,
            dropout=self.config.transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.transformer_dropout),
            nn.Linear(d_model // 2, self.config.n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, window_size, features)

        Returns:
            Tensor of shape (batch, n_classes) - logits
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer forward
        x = self.transformer(x)  # (batch, window_size, d_model)

        # Use center frame
        center_idx = x.size(1) // 2
        center_features = x[:, center_idx, :]

        # Classification
        logits = self.classifier(center_features)
        return logits


def create_boundary_detector(
    input_size: int,
    config: Optional[BoundaryDetectorConfig] = None
) -> nn.Module:
    """Factory function to create boundary detector model."""
    config = config or get_config().boundary_detector

    if config.model_type == "tcn":
        return BoundaryDetectorTCN(input_size, config)
    elif config.model_type == "transformer":
        return BoundaryDetectorTransformer(input_size, config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ==============================================================================
# CRF LAYER
# ==============================================================================

class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.

    Implements the forward algorithm for computing the partition function
    and the Viterbi algorithm for decoding.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        # Transition matrix: transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: Tensor,
        tags: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the negative log-likelihood loss.

        Args:
            emissions: Emission scores (batch, seq_len, num_tags)
            tags: Ground truth tags (batch, seq_len)
            mask: Sequence mask (batch, seq_len), 1 for valid positions

        Returns:
            Negative log-likelihood loss (scalar)
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        # Compute log-likelihood
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)

        # Return negative log-likelihood
        return -log_likelihood.mean()

    def decode(
        self,
        emissions: Tensor,
        mask: Optional[Tensor] = None
    ) -> List[List[int]]:
        """
        Decode best tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores (batch, seq_len, num_tags)
            mask: Sequence mask (batch, seq_len)

        Returns:
            List of best tag sequences
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        return self._viterbi_decode(emissions, mask)

    def _compute_log_likelihood(
        self,
        emissions: Tensor,
        tags: Tensor,
        mask: Tensor
    ) -> Tensor:
        """Compute log-likelihood of tag sequence."""
        batch_size, seq_len, _ = emissions.shape

        # Score of correct path
        score = self._compute_score(emissions, tags, mask)

        # Partition function (log-sum-exp of all paths)
        partition = self._compute_partition(emissions, mask)

        return score - partition

    def _compute_score(
        self,
        emissions: Tensor,
        tags: Tensor,
        mask: Tensor
    ) -> Tensor:
        """Compute score of given tag sequence."""
        batch_size, seq_len, _ = emissions.shape

        # Start transition
        score = self.start_transitions[tags[:, 0]]

        # Emission score for first tag
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            # Transition score
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            trans_score = self.transitions[curr_tags, prev_tags]

            # Emission score
            emit_score = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze(1)

            # Only add if position is valid
            score += (trans_score + emit_score) * mask[:, i].float()

        # End transition
        last_tags = tags.gather(1, mask.sum(1).long().unsqueeze(1) - 1).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Compute partition function using forward algorithm."""
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize with start transition + first emission
        alpha = self.start_transitions + emissions[:, 0]  # (batch, num_tags)

        for i in range(1, seq_len):
            # Broadcast alpha and add transition scores
            # alpha: (batch, num_tags) -> (batch, num_tags, 1)
            # transitions: (num_tags, num_tags) -> score of going from j to i
            broadcast_alpha = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)

            # Score: alpha[j] + trans[i, j] + emit[i]
            next_alpha = broadcast_alpha + self.transitions + broadcast_emissions

            # Log-sum-exp over previous tags
            next_alpha = torch.logsumexp(next_alpha, dim=1)  # (batch, num_tags)

            # Mask
            alpha = torch.where(mask[:, i].unsqueeze(1), next_alpha, alpha)

        # Add end transition
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(
        self,
        emissions: Tensor,
        mask: Tensor
    ) -> List[List[int]]:
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Best previous tag for each current tag
            next_score, indices = next_score.max(dim=1)

            # Mask
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition
        score = score + self.end_transitions

        # Backtrack
        best_tags_list = []
        seq_ends = mask.sum(dim=1) - 1

        for batch_idx in range(batch_size):
            best_tags = []
            seq_end = seq_ends[batch_idx].item()

            # Best last tag
            _, best_last_tag = score[batch_idx].max(dim=0)
            best_tags.append(best_last_tag.item())

            # Backtrack
            for hist in reversed(history[:int(seq_end)]):
                best_last_tag = hist[batch_idx][best_last_tag]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


# ==============================================================================
# REACH DETECTOR (BiLSTM-CRF)
# ==============================================================================

class ReachDetector(nn.Module):
    """
    Reach detector using BiLSTM with optional CRF layer.

    Performs sequence labeling to identify reach start/end frames.
    Tags: O (outside), B-reach (begin), I-reach (inside), E-reach (end)
    """

    def __init__(
        self,
        input_size: int,
        config: Optional[ReachDetectorConfig] = None
    ):
        super().__init__()
        self.config = config or get_config().reach_detector

        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        bidirectional = self.config.bidirectional
        num_tags = self.config.n_tags

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.config.dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Emission layer
        self.emission = nn.Linear(lstm_output_size, num_tags)
        self.dropout = nn.Dropout(self.config.dropout)

        # CRF layer
        self.use_crf = self.config.use_crf
        if self.use_crf:
            self.crf = CRF(num_tags)

    def forward(
        self,
        x: Tensor,
        tags: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List[List[int]]]]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, features)
            tags: Ground truth tags for training (batch, seq_len)
            mask: Sequence mask (batch, seq_len)

        Returns:
            If training (tags provided): loss tensor
            If inference: (emissions, decoded_tags)
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = self.dropout(lstm_out)

        # Emission scores
        emissions = self.emission(lstm_out)  # (batch, seq_len, num_tags)

        if tags is not None:
            # Training: compute loss
            if self.use_crf:
                loss = self.crf(emissions, tags, mask)
            else:
                # Standard cross-entropy loss
                loss = F.cross_entropy(
                    emissions.view(-1, emissions.size(-1)),
                    tags.view(-1),
                    reduction='mean'
                )
            return loss
        else:
            # Inference: decode
            if self.use_crf:
                decoded = self.crf.decode(emissions, mask)
            else:
                decoded = emissions.argmax(dim=-1).tolist()
            return emissions, decoded

    def decode(self, x: Tensor, mask: Optional[Tensor] = None) -> List[List[int]]:
        """Decode tag sequence."""
        _, decoded = self.forward(x, tags=None, mask=mask)
        return decoded


# ==============================================================================
# OUTCOME CLASSIFIER (MLP)
# ==============================================================================

class OutcomeClassifier(nn.Module):
    """
    Outcome classifier using Multi-Layer Perceptron.

    Takes aggregated features from a reach segment and predicts outcome class.
    """

    def __init__(
        self,
        input_size: int,
        config: Optional[OutcomeClassifierConfig] = None
    ):
        super().__init__()
        self.config = config or get_config().outcome_classifier

        layers = []
        in_features = input_size

        for hidden_size in self.config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))

            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout))

            in_features = hidden_size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, self.config.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Aggregated features (batch, features)

        Returns:
            Logits (batch, n_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

class ModelFactory:
    """Factory for creating models."""

    @staticmethod
    def create_boundary_detector(
        input_size: int,
        config: Optional[Config] = None
    ) -> nn.Module:
        """Create boundary detector model."""
        config = config or get_config()
        return create_boundary_detector(input_size, config.boundary_detector)

    @staticmethod
    def create_reach_detector(
        input_size: int,
        config: Optional[Config] = None
    ) -> nn.Module:
        """Create reach detector model."""
        config = config or get_config()
        return ReachDetector(input_size, config.reach_detector)

    @staticmethod
    def create_outcome_classifier(
        input_size: int,
        config: Optional[Config] = None
    ) -> nn.Module:
        """Create outcome classifier model."""
        config = config or get_config()
        return OutcomeClassifier(input_size, config.outcome_classifier)


# ==============================================================================
# MODEL SUMMARY UTILITIES
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> None:
    """Print model summary."""
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("\nArchitecture:")
    print(model)


# ==============================================================================
# MAIN / TESTING
# ==============================================================================

if __name__ == "__main__":
    from config import get_config

    config = get_config()

    print("=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)

    # Test dimensions
    batch_size = 8
    window_size = 61
    seq_len = 120
    n_features = 19  # From feature extraction

    # Test Boundary Detector (TCN)
    print("\n1. Boundary Detector (TCN)")
    boundary_model_tcn = BoundaryDetectorTCN(n_features, config.boundary_detector)
    x = torch.randn(batch_size, window_size, n_features)
    out = boundary_model_tcn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(boundary_model_tcn):,}")

    # Test Boundary Detector (Transformer)
    print("\n2. Boundary Detector (Transformer)")
    config.boundary_detector.model_type = "transformer"
    boundary_model_trans = BoundaryDetectorTransformer(n_features, config.boundary_detector)
    out = boundary_model_trans(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(boundary_model_trans):,}")

    # Test Reach Detector (BiLSTM-CRF)
    print("\n3. Reach Detector (BiLSTM-CRF)")
    reach_model = ReachDetector(n_features, config.reach_detector)
    x_seq = torch.randn(batch_size, seq_len, n_features)
    tags = torch.randint(0, 4, (batch_size, seq_len))

    # Training mode
    loss = reach_model(x_seq, tags=tags)
    print(f"   Input shape: {x_seq.shape}")
    print(f"   Tags shape: {tags.shape}")
    print(f"   Training loss: {loss.item():.4f}")

    # Inference mode
    emissions, decoded = reach_model(x_seq)
    print(f"   Emissions shape: {emissions.shape}")
    print(f"   Decoded length: {len(decoded)}, first seq length: {len(decoded[0])}")
    print(f"   Parameters: {count_parameters(reach_model):,}")

    # Test Outcome Classifier (MLP)
    print("\n4. Outcome Classifier (MLP)")
    # Aggregated features: 10 stats * n_features
    aggregated_size = 10 * n_features
    outcome_model = OutcomeClassifier(aggregated_size, config.outcome_classifier)
    x_agg = torch.randn(batch_size, aggregated_size)
    out = outcome_model(x_agg)
    print(f"   Input shape: {x_agg.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(outcome_model):,}")

    # Test CRF layer independently
    print("\n5. CRF Layer")
    crf = CRF(num_tags=4)
    emissions = torch.randn(batch_size, seq_len, 4)
    tags = torch.randint(0, 4, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    loss = crf(emissions, tags, mask)
    decoded = crf.decode(emissions, mask)
    print(f"   CRF loss: {loss.item():.4f}")
    print(f"   Decoded sequences: {len(decoded)}")

    print("\n" + "=" * 60)
    print("Model architecture tests complete")
    print("=" * 60)
