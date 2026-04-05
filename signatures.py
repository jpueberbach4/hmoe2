import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import signatory
except ImportError:
    signatory = None


class SignatureBackend(nn.Module):
    """Signature-based backend for geometric sequence features.

    Computes the streamed path signature of an input sequence and projects
    it into a lower-dimensional hidden space via an MLP. The signature
    captures ordering and higher-order interactions in a time-warp invariant way.

    Attributes:
        depth (int): Truncation depth of the signature transform.
        sig_channels (int): Number of signature output channels.
        net (nn.Sequential): Projection network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 2,
        dropout_p: float = 0.2
    ):
        """Initializes the SignatureBackend.

        Args:
            input_dim (int): Number of input features per timestep.
            hidden_dim (int): Output hidden dimension.
            depth (int, optional): Signature truncation depth. Defaults to 2.
            dropout_p (float, optional): Dropout probability. Defaults to 0.2.

        Raises:
            ImportError: If `signatory` is not installed.
        """
        super().__init__()

        # Ensure dependency is available
        if signatory is None:
            raise ImportError(
                "The 'signatory' library is required to use the SignatureBackend. "
                "Please run: pip install signatory"
            )

        # Store configuration
        self.depth = depth

        # Compute signature output dimensionality
        self.sig_channels = signatory.signature_channels(input_dim, depth)

        # Projection MLP
        self.net = nn.Sequential(
            nn.Linear(self.sig_channels, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies streaming signature transform and projection.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, hidden_dim).
        """
        # Compute streaming signature (per timestep, causal)
        sig_path = signatory.signature(
            x,
            depth=self.depth,
            stream=True,
            basepoint=True
        )

        # Project to hidden space
        return self.net(sig_path)