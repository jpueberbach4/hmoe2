import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Dict

# Map signature as a backend (so it can get imported from backends)
from hmoe2.signatures import SignatureBackend
from hmoe2.motifs import MotifsBackend

class StrictCausalConv1d(nn.Module):
    """Implements a strictly causal 1D convolution layer.

    This layer ensures that the convolution operation does not incorporate
    any future information (i.e., no lookahead bias). It achieves this by
    applying symmetric padding and then removing the extra elements that
    correspond to future timesteps.

    Attributes:
        causal_padding (int): Amount of padding applied to enforce causality.
        conv (nn.Conv1d): Underlying convolutional layer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        """Initializes the StrictCausalConv1d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            dilation (int): Dilation factor for the convolution.
        """
        super().__init__()

        # Compute the amount of padding required so that the convolution
        # only depends on current and past inputs (causal structure)
        self.causal_padding = (kernel_size - 1) * dilation

        # Define a standard Conv1d layer with symmetric padding applied
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the causal convolution to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Sequence].

        Returns:
            torch.Tensor: Output tensor with strictly causal structure.
        """
        # Apply convolution with symmetric padding
        out = self.conv(x)

        # Remove the trailing elements that correspond to "future" context
        # introduced by padding, ensuring strict causality
        return out[:, :, :-self.causal_padding] if self.causal_padding > 0 else out


class LinearBackend(nn.Module):
    """Feedforward backend with no temporal memory.

    This module processes each timestep independently using fully connected
    layers. It is suitable for scenarios where no sequential dependency is required.

    Attributes:
        net (nn.Sequential): Sequential stack of linear, activation, and normalization layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the LinearBackend module.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of hidden representation.
        """
        super().__init__()

        # Config
        dropout_p = config.get('dropout', 0.2)
        # Define a simple feedforward network with normalization and activation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the input tensor through the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, Sequence, Hidden].
        """
        # Apply the feedforward network independently across timesteps
        return self.net(x)


class TcnBackend(nn.Module):
    """Temporal Convolutional Network backend with residual connections.

    This module applies a stack of dilated causal convolutions to capture
    temporal dependencies at multiple scales, with residual connections
    to stabilize training.

    Attributes:
        convs (nn.ModuleList): List of causal convolution layers.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the TcnBackend module.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of hidden representation.
            dilations (list): List of dilation factors for each layer.
            dropout_p (float): Dropout probability.
        """
        super().__init__()

        # Config
        dilations = config.get('dilations', [1,2,4,8])
        dropout_p = config.get('dropout', 0.2)
  
        # Initialize a list to hold convolutional layers
        self.convs = nn.ModuleList()

        # First layer maps input dimension to hidden dimension
        self.convs.append(StrictCausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=dilations[0]))

        # Subsequent layers maintain hidden dimension
        for d in dilations[1:]:
            self.convs.append(StrictCausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=d))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the input through the TCN layers.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, Sequence, Hidden].
        """
        # Transpose to match Conv1d expected input format
        x = x.transpose(1, 2)

        # Apply first convolution layer (dimension-changing)
        x = self.convs[0](x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Apply remaining layers with residual connections
        for conv in self.convs[1:]:
            residual = x
            x = conv(x)
            x = F.gelu(x)
            x = self.dropout(x)

            # Add residual connection to preserve gradient flow
            x = x + residual

        # Transpose back to original format
        return x.transpose(1, 2)


class GruBackend(nn.Module):
    """GRU-based sequential model for temporal dependencies.

    This module uses a multi-layer GRU to capture sequential patterns,
    followed by dropout for regularization.

    Attributes:
        gru (nn.GRU): GRU layer.
        output_dropout (nn.Dropout): Dropout applied to outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the GruBackend module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of GRU layers.
            dropout_p (float): Dropout probability.
        """
        super().__init__()

        # Config
        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        # Initialize GRU with optional inter-layer dropout
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Dropout applied to GRU outputs
        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the input through the GRU.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        # Pass input through GRU
        out, _ = self.gru(x)

        # Apply dropout to reduce overfitting
        out = self.output_dropout(out)

        return out


class CausalTransformerBackend(nn.Module):
    """Transformer encoder with causal masking and sinusoidal embeddings.

    This module applies a Transformer encoder while enforcing causality
    via masking and augmenting inputs with positional encodings.

    Attributes:
        hidden_dim (int): Hidden dimension size.
        input_proj (nn.Linear): Input projection layer.
        transformer (nn.TransformerEncoder): Transformer encoder stack.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the CausalTransformerBackend module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden representation dimension.
            num_layers (int): Number of transformer layers.
            nheads (int): Number of attention heads.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        
        # Config
        num_layers = config.get('num_layers', 2)
        nheads = config.get('nheads', 4)
        dropout_p = config.get('dropout', 0.2)

        self.hidden_dim = hidden_dim

        # Linear projection to match transformer dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Define transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def _get_sinusoidal_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates sinusoidal positional embeddings.

        Args:
            seq_len (int): Sequence length.
            device (torch.device): Device for tensor allocation.

        Returns:
            torch.Tensor: Positional embeddings of shape [1, Seq, Hidden].
        """
        # Generate position indices
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)

        # Compute scaling factors for sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float() *
            (-math.log(10000.0) / self.hidden_dim)
        ).to(device)

        # Initialize positional encoding tensor
        pe = torch.zeros(1, seq_len, self.hidden_dim, device=device)

        # Apply sine to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through the transformer encoder.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        seq_len = x.size(1)

        # Project input and add positional encoding
        x = self.input_proj(x)
        x = x + self._get_sinusoidal_embeddings(seq_len, x.device)

        # Generate causal mask to prevent attention to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Apply transformer encoder with causal masking
        return self.transformer(x, mask=causal_mask, is_causal=True)


class GatedResidualBackend(nn.Module):
    """Feedforward network with gated residual connections.

    This module uses a Gated Linear Unit (GLU) to dynamically control
    feature flow, combined with residual connections and normalization.

    Attributes:
        input_proj (nn.Linear): Input projection layer.
        net (nn.Sequential): Feedforward transformation layers.
        dropout (nn.Dropout): Dropout layer.
        layer_norm (nn.LayerNorm): Normalization layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the GatedResidualBackend module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden representation dimension.
            dropout_p (float): Dropout probability.
        """
        super().__init__()

        # Config
        dropout_p = config.get('dropout', 0.2)

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Define feedforward network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input with gated residual connections.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        # Project input for residual connection
        residual = self.input_proj(x)

        # Apply feedforward transformation
        x = self.net(residual)
        x = self.dropout(x)

        # Apply GLU to split and gate features
        x = F.glu(x, dim=-1)

        # Combine with residual and normalize
        return self.layer_norm(x + residual)


class LstmBackend(nn.Module):
    """LSTM-based sequential model for long-term dependencies.

    This module uses an LSTM network to model long-range temporal patterns,
    followed by dropout for regularization.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
        output_dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the LstmBackend module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of LSTM layers.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        
        # Config
        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        # Initialize LSTM with optional dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Output dropout layer
        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through the LSTM.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        # Apply LSTM and discard hidden/cell states
        out, _ = self.lstm(x)

        # Apply dropout to outputs
        out = self.output_dropout(out)

        return out
    
class RnnBackend(nn.Module):
    """Vanilla RNN-based sequential model.
    
    This module uses a standard Elman Recurrent Neural Network. It is computationally 
    lighter than GRU or LSTM, making it highly efficient for short-term sequential 
    dependencies, though it may struggle with long-range vanishing gradients.

    Attributes:
        rnn (nn.RNN): The vanilla RNN layer.
        output_dropout (nn.Dropout): Dropout applied to outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        """Initializes the RnnBackend module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of RNN layers.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        
        # Config
        num_layers = config.get('num_layers', 2)
        dropout_p = config.get('dropout', 0.2)

        # Initialize Vanilla RNN with optional inter-layer dropout
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Output dropout layer
        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through the vanilla RNN.

        Args:
            x (torch.Tensor): Input tensor [Batch, Sequence, Features].

        Returns:
            torch.Tensor: Output tensor [Batch, Sequence, Hidden].
        """
        # Apply RNN and discard the final hidden state
        out, _ = self.rnn(x)

        # Apply dropout to outputs
        out = self.output_dropout(out)

        return out
