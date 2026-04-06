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
        """Processes the input path through a Rolling Window Signature transform.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        b, s, c = x.size()
        
        # The lookback window: Calculate the geometry of the last 60 candles.
        # (You can move this to __init__ if you want it configurable)
        window_length = 40
        
        # 1. Strict Causal Padding
        # Pad the left side so the first 60 candles don't look into the future
        x_t = x.transpose(1, 2)
        x_padded = F.pad(x_t, (window_length - 1, 0), mode="replicate")
        
        # 2. Extract Sliding Windows
        # Output shape: [Batch, Channels, Sequence, Window]
        windows = x_padded.unfold(dimension=2, size=window_length, step=1)
        
        # 3. Reshape for the Signatory Engine
        # Signatory expects [Batch, Seq, Channels]. 
        # We flatten our batch and sequence dimensions to process all windows in parallel.
        windows = windows.permute(0, 2, 3, 1).contiguous() # [Batch, Seq, Window, Channels]
        flat_windows = windows.view(b * s, window_length, c)
        
        # 4. Calculate localized Signatures!
        # Notice stream=True is REMOVED. We are calculating the fixed signature 
        # of the 60-candle window, not an infinite stream.
        sig_path = signatory.signature(flat_windows, depth=self.depth, basepoint=True)
        
        # 5. Reshape back to expected output and project
        sig_path = sig_path.view(b, s, -1)
        
        return self.net(sig_path)