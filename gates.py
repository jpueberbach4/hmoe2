import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from hmoe2.tensor import HmoeTensor
from hmoe2.backends import StrictCausalConv1d


class HmoeGate(nn.Module):
    """Stateless routing gate using a simple linear projection.

    This gate computes routing probabilities independently for each timestep
    without considering temporal dependencies. It is suitable for baseline
    or purely reactive routing decisions.

    Attributes:
        routing_head (nn.Linear): Linear layer producing routing logits.
        noise_std (float): Standard deviation of optional noise (disabled by default).
    """

    def __init__(self, input_dim: int, num_children: int):
        """Initializes the HmoeGate.

        Args:
            input_dim (int): Number of input features.
            num_children (int): Number of child nodes (experts) to route to.
        """
        super().__init__()

        # Linear projection mapping input features to routing logits
        self.routing_head = nn.Linear(input_dim, num_children)

        # Noise level for exploration (set to zero for deterministic behavior)
        self.noise_std = 0.0

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Computes routing weights for each child.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Routing weights of shape [Batch, Sequence, num_children].
        """
        # Convert structured payload into raw tensor format
        raw_data = payload_tensor.to_tensor()

        # Compute routing logits for each child node
        routing_logits = self.routing_head(raw_data)

        # Normalize logits into probabilities using softmax
        routing_weights = F.softmax(routing_logits, dim=-1)

        return routing_weights


class HmoeGateTCN(nn.Module):
    """Temporal routing gate using dilated causal convolutions.

    This gate leverages a Temporal Convolutional Network (TCN) structure
    to capture both short-term and long-term temporal dependencies before
    making routing decisions.

    Attributes:
        noise_std (float): Standard deviation of noise for exploration.
        conv_stack (nn.ModuleList): Stack of causal convolution layers.
        dropout (nn.Dropout): Dropout layer for regularization.
        routing_head (nn.Linear): Linear layer mapping hidden states to routing logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_children: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        noise_std: float = 0.1,
        dropout_p: float = 0.1
    ):
        """Initializes the HmoeGateTCN.

        Args:
            input_dim (int): Number of input features.
            num_children (int): Number of routing targets.
            hidden_dim (int): Hidden representation size.
            kernel_size (int): Convolution kernel size.
            dilations (List[int]): Dilation factors for each layer.
            noise_std (float): Noise level for stochastic routing.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        # Store noise level for optional stochastic exploration
        self.noise_std = noise_std

        # Initialize stack of causal convolution layers
        self.conv_stack = nn.ModuleList()

        # First layer transforms input dimension into hidden dimension
        self.conv_stack.append(
            StrictCausalConv1d(input_dim, hidden_dim, kernel_size, dilation=dilations[0])
        )

        # Remaining layers preserve hidden dimension while increasing receptive field
        for d in dilations[1:]:
            self.conv_stack.append(
                StrictCausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=d)
            )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_p)

        # Final linear layer mapping hidden states to routing logits
        self.routing_head = nn.Linear(hidden_dim, num_children)

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Computes routing weights using temporal convolution.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Routing weights of shape [Batch, Sequence, num_children].
        """
        # Convert structured payload into tensor
        raw_data = payload_tensor.to_tensor()

        # Rearrange dimensions to match Conv1d input format
        x = raw_data.transpose(1, 2)

        # Apply first convolution layer (dimension transformation)
        x = self.conv_stack[0](x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Apply remaining layers with residual connections
        for conv in self.conv_stack[1:]:
            residual = x
            x = conv(x)
            x = F.gelu(x)
            x = self.dropout(x)

            # Add residual connection to preserve temporal information
            x = x + residual

        # Restore original tensor layout
        x = x.transpose(1, 2)

        # Compute routing logits from processed features
        routing_logits = self.routing_head(x)

        # Optionally inject noise during training for exploration
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise

        # Convert logits into probability distribution
        routing_weights = F.softmax(routing_logits, dim=-1)

        return routing_weights


class HmoeGateTopK(nn.Module):
    """Sparse routing gate using Top-K selection.

    This gate enforces sparsity by selecting only the top-K most confident
    routing logits and masking all others, encouraging expert specialization.

    Attributes:
        routing_head (nn.Linear): Linear layer producing routing logits.
        k (int): Number of top experts to select.
        noise_std (float): Noise level for stochastic exploration.
    """

    def __init__(self, input_dim: int, num_children: int, k: int = 1, noise_std: float = 0.1):
        """Initializes the HmoeGateTopK.

        Args:
            input_dim (int): Number of input features.
            num_children (int): Number of routing targets.
            k (int): Number of top experts to retain.
            noise_std (float): Noise level for stochastic routing.
        """
        super().__init__()

        # Linear projection for routing logits
        self.routing_head = nn.Linear(input_dim, num_children)

        # Ensure k does not exceed number of children
        self.k = min(k, num_children)

        # Noise level for exploration
        self.noise_std = noise_std

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Computes sparse routing weights using Top-K masking.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Sparse routing weights.
        """
        # Convert payload to tensor
        raw_data = payload_tensor.to_tensor()

        # Compute routing logits
        routing_logits = self.routing_head(raw_data)

        # Optionally add noise during training
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise

        # Select top-K logits and their indices
        topk_logits, topk_indices = torch.topk(routing_logits, self.k, dim=-1)

        # Initialize mask with negative infinity
        mask = torch.full_like(routing_logits, float('-inf'))

        # Place top-K logits into the mask at their respective indices
        mask.scatter_(-1, topk_indices, topk_logits)

        # Apply softmax; masked values become zero probability
        routing_weights = F.softmax(mask, dim=-1)

        return routing_weights


class HmoeGateGRU(nn.Module):
    """Stateful routing gate using GRU for temporal modeling.

    This gate leverages a GRU to track temporal dynamics and generate
    routing decisions based on sequential context.

    Attributes:
        noise_std (float): Noise level for exploration.
        gru (nn.GRU): GRU module for sequence modeling.
        routing_head (nn.Linear): Linear layer mapping hidden states to routing logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_children: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout_p: float = 0.1,
        noise_std: float = 0.0
    ):
        """Initializes the HmoeGateGRU.

        Args:
            input_dim (int): Input feature dimension.
            num_children (int): Number of routing targets.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of GRU layers.
            dropout_p (float): Dropout probability.
            noise_std (float): Noise level for stochastic routing.
        """
        super().__init__()

        # Store noise level for optional stochasticity
        self.noise_std = noise_std

        # Initialize GRU for temporal feature extraction
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Linear projection from hidden state to routing logits
        self.routing_head = nn.Linear(hidden_dim, num_children)

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Computes routing weights using GRU-based temporal context.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Routing weights.
        """
        # Convert payload into raw tensor format
        raw_data = payload_tensor.to_tensor()

        # Process sequence through GRU to capture temporal dependencies
        out, _ = self.gru(raw_data)

        # Compute routing logits from GRU outputs
        routing_logits = self.routing_head(out)

        # Optionally add noise during training
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise

        # Normalize logits into probabilities
        routing_weights = F.softmax(routing_logits, dim=-1)

        return routing_weights