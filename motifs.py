import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class MotifsBackend(nn.Module):
    """Motif-matching backend using sliding Pearson correlation.

    Learns a set of temporal motifs and computes their similarity against
    all sliding windows of the input sequence. Similarity is measured via
    per-feature Pearson correlation, followed by nonlinear gating and a
    feedforward projection.

    Attributes:
        num_motifs (int): Number of learnable motifs.
        motif_length (int): Temporal length of each motif.
        input_dim (int): Number of input features.
        motifs (nn.Parameter): Learnable motif tensor
            of shape (num_motifs, input_dim, motif_length).
        net (nn.Sequential): Projection network.
        _forward_calls (int): Forward pass counter (debugging/monitoring).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        config: Dict = {}
    ):
        """Initializes the MotifsBackend.

        Args:
            input_dim (int): Number of input channels.
            hidden_dim (int): Output hidden dimension.
            num_motifs (int, optional): Number of motifs. Defaults to 8.
            motif_length (int, optional): Length of each motif. Defaults to 12.
            dropout_p (float, optional): Dropout probability. Defaults to 0.2.
        """
        super().__init__()

        # Config
        num_motifs = config.get('num_motifs', 8)
        motif_length = config.get('motif_length', 12)
        dropout_p = config.get('dropout', 0.2)

        # Store configuration
        self.num_motifs = num_motifs
        self.motif_length = motif_length
        self.input_dim = input_dim

        # Learnable motifs: (num_motifs, input_dim, motif_length)
        self.motifs = nn.Parameter(
            torch.randn(num_motifs, input_dim, motif_length)
        )

        # Projection MLP
        self.net = nn.Sequential(
            nn.Linear(num_motifs * input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Forward call counter
        self._forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes motif similarity features.

        Pipeline:
            1. Causal padding + sliding window extraction
            2. Z-normalization
            3. Pearson correlation (per feature)
            4. Nonlinear gating
            5. Flatten + projection

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, hidden_dim).
        """
        # Input dimensions
        b, s, c = x.size()

        # Convert to (batch, channels, sequence)
        x_t = x.transpose(1, 2)

        # Causal left padding (no future leakage)
        x_padded = F.pad(
            x_t,
            (self.motif_length - 1, 0),
            mode="replicate"
        )

        # Sliding windows: (batch, channels, sequence, motif_length)
        windows = x_padded.unfold(
            dimension=2,
            size=self.motif_length,
            step=1
        )

        # Z-normalize windows
        w_mean = windows.mean(dim=-1, keepdim=True)
        w_std = windows.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
        windows_z = (windows - w_mean) / w_std

        # Z-normalize motifs
        m_mean = self.motifs.mean(dim=-1, keepdim=True)
        m_std = self.motifs.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
        motifs_z = (self.motifs - m_mean) / m_std

        # Pearson correlation via einsum -> (batch, sequence, motif, channel)
        sim_per_feature = torch.einsum(
            "bcsl,mcl->bsmc",
            windows_z,
            motifs_z
        ) / self.motif_length

        # Gate: keep positive correlations, amplify strong matches
        gated_sim = torch.pow(F.relu(sim_per_feature), 4)

        # Track forward calls
        self._forward_calls += 1

        # Flatten motif + channel dims
        flat_profiles = gated_sim.contiguous().reshape(
            b,
            s,
            self.num_motifs * self.input_dim
        )

        # Project to hidden space
        return self.net(flat_profiles)