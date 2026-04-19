import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

from hmoe2.signatures import SignatureBackend
from hmoe2.motifs import MotifsBackend
from hmoe2.snn import SnnBackend


class StrictCausalConv1d(nn.Module):
    """Strictly causal 1D convolution using explicit left padding.

    Ensures that output at time t only depends on inputs <= t.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of convolution kernel.
        dilation (int): Dilation factor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))  # left-pad only

        return self.conv(x)


class LinearBackend(nn.Module):
    """Feedforward backend without temporal modeling.

    Implements a symmetric pre-norm MLP with GELU activations.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden/output dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}
        dropout_p = config.get('dropout', 0.2)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, T, D] or [B, D].

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.net(x)


class TcnBackend(nn.Module):
    """Temporal Convolutional Network with causal convolutions.

    Uses dilated convolutions with residual connections and pre-normalization.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden/output dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}

        dilations = config.get('dilations', [1, 2, 4, 8])
        dropout_p = config.get('dropout', 0.2)
        kernel_size = config.get('kernel_size', 3)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer maps input_dim -> hidden_dim
        self.convs.append(
            StrictCausalConv1d(input_dim, hidden_dim, kernel_size, dilations[0])
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Residual blocks keep hidden_dim constant
        for d in dilations[1:]:
            self.convs.append(
                StrictCausalConv1d(hidden_dim, hidden_dim, kernel_size, d)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(p=dropout_p)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, T, D].

        Returns:
            torch.Tensor: Output tensor [B, T, D].
        """
        # First convolution (no residual)
        x_conv = self.convs[0](x.transpose(1, 2)).transpose(1, 2)
        x = self.norms[0](x_conv)
        x = F.gelu(x)
        x = self.dropout(x)

        # Residual blocks
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            residual = x

            x_conv = conv(x.transpose(1, 2)).transpose(1, 2)
            x = norm(x_conv)
            x = F.gelu(x)
            x = self.dropout(x)

            x = x + residual  # residual connection

        return self.final_norm(x)


class GruBackend(nn.Module):
    """GRU-based sequential model.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}

        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, T, D].

        Returns:
            torch.Tensor: Output tensor [B, T, H].
        """
        out, _ = self.gru(x)
        return self.output_dropout(out)


class CausalTransformerBackend(nn.Module):
    """Transformer encoder with causal masking and sinusoidal embeddings.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Model dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}

        num_layers = config.get('num_layers', 2)
        nheads = config.get('nheads', 4)
        dropout_p = config.get('dropout', 0.2)

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_sinusoidal_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional encodings.

        Args:
            seq_len (int): Sequence length.
            device (torch.device): Target device.

        Returns:
            torch.Tensor: Positional encodings [1, T, D].
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float() *
            (-math.log(10000.0) / self.hidden_dim)
        ).to(device)

        pe = torch.zeros(1, seq_len, self.hidden_dim, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, T, D].

        Returns:
            torch.Tensor: Output tensor [B, T, H].
        """
        seq_len = x.size(1)

        x = self.input_proj(x)
        x = x + self._get_sinusoidal_embeddings(seq_len, x.device)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )

        return self.transformer(x, mask=causal_mask)


class GatedResidualBackend(nn.Module):
    """Feedforward network with gated residual connections (GLU).

    Uses pre-normalization and a gated linear unit for modulation.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden/output dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}
        dropout_p = config.get('dropout', 0.2)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, T, D].

        Returns:
            torch.Tensor: Output tensor [B, T, H].
        """
        residual = self.input_proj(x)  # match dimensions

        branch = self.layer_norm(residual)  # pre-norm
        branch = self.net(branch)
        branch = F.glu(branch, dim=-1)  # gating

        return residual + branch  # residual addition


class LstmBackend(nn.Module):
    """LSTM-based sequential model.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}

        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out, _ = self.lstm(x)
        return self.output_dropout(out)


class RnnBackend(nn.Module):
    """Vanilla RNN-based sequential model.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        config (Dict, optional): Configuration dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = None):
        super().__init__()
        config = config or {}

        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out, _ = self.rnn(x)
        return self.output_dropout(out)