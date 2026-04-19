import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

try:
    import signatory
except ImportError:
    signatory = None


class SignatureBackend(nn.Module):
    """Logsignature-based sequence encoder with time augmentation.

    This backend:
    - Extracts sliding windows from the input sequence
    - Augments each window with a normalized time dimension
    - Computes logsignatures (capturing path geometry)
    - Projects the resulting features into a hidden space

    Key advantages:
    - Captures higher-order temporal interactions (beyond standard RNN/Conv)
    - Encodes path-dependent structure (ordering matters)
    - Time augmentation enables velocity/momentum representation

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Output feature dimension.
        config (Dict): Configuration dictionary.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        config: Dict = {}
    ):
        super().__init__()

        # Configuration parameters
        self.depth = config.get('depth', 3)  # logsignature truncation depth
        dropout_p = config.get('dropout', 0.2)
        self.window_length = config.get('window_length', 60)

        # Ensure dependency is available
        if signatory is None:
            raise ImportError(
                "The 'signatory' library is required to use the SignatureBackend. "
                "Please run: pip install signatory"
            )

        # Add 1 dimension for time augmentation
        self.augmented_dim = input_dim + 1

        # Compute logsignature feature size for augmented input
        self.sig_channels = signatory.logsignature_channels(
            self.augmented_dim,
            self.depth
        )

        # Projection network to map logsignature -> hidden space
        self.net = nn.Sequential(
            nn.Linear(self.sig_channels, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logsignature features for input sequence.

        Args:
            x (torch.Tensor): Input tensor [B, T, C].

        Returns:
            torch.Tensor: Encoded tensor [B, T, hidden_dim].
        """
        # Extract dimensions
        b, s, c = x.size()
        device = x.device
        window_length = self.window_length

        # ---------------------------------------------------------------------
        # 1. CAUSAL PADDING
        # ---------------------------------------------------------------------
        # Convert to [B, C, T] for Conv-style operations
        x_t = x.transpose(1, 2)

        # Pad left so each timestep has a full window (strictly causal)
        # replicate padding preserves boundary values
        x_padded = F.pad(
            x_t,
            (window_length - 1, 0),
            mode="replicate"
        )

        # ---------------------------------------------------------------------
        # 2. SLIDING WINDOW EXTRACTION
        # ---------------------------------------------------------------------
        # Extract overlapping windows along time dimension
        # Output: [B, C, T, window_length]
        windows = x_padded.unfold(
            dimension=2,
            size=window_length,
            step=1
        )

        # Rearrange to [B, T, window_length, C]
        windows = windows.permute(0, 2, 3, 1).contiguous()

        # Flatten batch and sequence for efficient processing
        # Shape: [(B*T), window_length, C]
        flat_windows = windows.view(b * s, window_length, c)

        # ---------------------------------------------------------------------
        # 3. TIME AUGMENTATION
        # ---------------------------------------------------------------------
        # Create normalized time vector [0, 1]
        t = torch.linspace(
            0,
            1,
            steps=window_length,
            device=device
        )

        # Expand to match batch*sequence dimension
        # Shape: [(B*T), window_length, 1]
        t = t.view(1, window_length, 1).expand(b * s, -1, -1)

        # Concatenate time with features
        # Shape: [(B*T), window_length, C+1]
        augmented_windows = torch.cat([t, flat_windows], dim=-1)

        # ---------------------------------------------------------------------
        # 4. LOGSIGNATURE COMPUTATION
        # ---------------------------------------------------------------------
        # Computes path signature in log-space
        # Captures geometric structure of the sequence path
        sig_path = signatory.logsignature(
            augmented_windows,
            depth=self.depth,
            basepoint=True  # includes starting reference point
        )

        # ---------------------------------------------------------------------
        # 5. RESHAPE + PROJECTION
        # ---------------------------------------------------------------------
        # Restore original batch/sequence structure
        sig_path = sig_path.view(b, s, -1)

        # Project to hidden dimension
        return self.net(sig_path)