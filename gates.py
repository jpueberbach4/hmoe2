import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from hmoe2.tensor import HmoeTensor
from hmoe2.backends import StrictCausalConv1d


class HmoeGate(nn.Module):
    """Stateless routing gate with dynamic load balancing.

    Uses a linear projection to produce routing logits and applies a
    dynamically updated bias term to encourage balanced expert utilization.

    Args:
        input_dim (int): Input feature dimension.
        num_children (int): Number of experts (routing targets).
    """

    def __init__(self, input_dim: int, num_children: int):
        super().__init__()

        # Linear projection from input features -> expert logits
        self.routing_head = nn.Linear(input_dim, num_children)

        # Optional Gaussian noise for exploration during training
        self.noise_std = 0.0

        # Running estimate of how much each expert is used (EMA)
        self.register_buffer(
            'expert_utilization',
            torch.ones(num_children) / num_children
        )

        # Bias added to logits to correct imbalance
        self.register_buffer(
            'dynamic_bias',
            torch.zeros(num_children)
        )

        # Target uniform utilization
        self.target_utilization = 1.0 / num_children

        # Learning rate for bias updates
        self.bias_lr = 0.01

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Compute routing weights.

        Args:
            payload_tensor (HmoeTensor): Input wrapper containing tensor data.

        Returns:
            torch.Tensor: Routing weights [B, T, num_children].
        """
        # Extract raw tensor [B, T, D]
        raw_data = payload_tensor.to_tensor()

        # Compute raw routing logits
        routing_logits = self.routing_head(raw_data)

        # Add dynamic bias (outside gradient flow of routing_head)
        routing_logits = routing_logits + self.dynamic_bias

        # Inject noise for exploration (only during training)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise

        # Convert logits -> probabilities
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Update utilization statistics and bias (no gradient)
        if self.training:
            # Average usage per expert across batch + time
            batch_util = routing_weights.mean(dim=(0, 1)).detach()

            # Exponential moving average update
            self.expert_utilization = (
                0.99 * self.expert_utilization + 0.01 * batch_util
            )

            # Compute deviation from target utilization
            util_error = self.target_utilization - self.expert_utilization

            # Adjust bias to encourage underused experts
            self.dynamic_bias = self.dynamic_bias + self.bias_lr * util_error

        return routing_weights


class HmoeGateTCN(nn.Module):
    """Temporal routing gate using causal convolutions.

    Extracts temporal features using a TCN before computing routing logits.
    Includes dynamic bias for load balancing.

    Args:
        input_dim (int): Input feature dimension.
        num_children (int): Number of experts.
        hidden_dim (int): Hidden feature dimension.
        kernel_size (int): Convolution kernel size.
        dilations (List[int]): Dilation schedule.
        noise_std (float): Noise std for exploration.
        dropout_p (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_children: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        noise_std: float = 0.0,
        dropout_p: float = 0.1
    ):
        super().__init__()

        self.noise_std = noise_std

        # Stack of causal convolutions (TCN)
        self.conv_stack = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.conv_stack.append(
            StrictCausalConv1d(input_dim, hidden_dim, kernel_size, dilation=dilations[0])
        )

        # Subsequent residual layers
        for d in dilations[1:]:
            self.conv_stack.append(
                StrictCausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=d)
            )

        self.dropout = nn.Dropout(p=dropout_p)

        # Final projection to expert logits
        self.routing_head = nn.Linear(hidden_dim, num_children)

        # Load balancing buffers
        self.register_buffer('expert_utilization', torch.ones(num_children) / num_children)
        self.register_buffer('dynamic_bias', torch.zeros(num_children))
        self.target_utilization = 1.0 / num_children
        self.bias_lr = 0.01

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Compute routing weights with temporal context.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Routing weights [B, T, num_children].
        """
        # Convert to [B, C, T] for Conv1d
        x = payload_tensor.to_tensor().transpose(1, 2)

        # First convolution (no residual)
        x = self.conv_stack[0](x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Residual TCN blocks
        for conv in self.conv_stack[1:]:
            residual = x  # save input for skip connection

            x = conv(x)
            x = F.gelu(x)
            x = self.dropout(x)

            x = x + residual  # residual addition

        # Back to [B, T, D]
        x = x.transpose(1, 2)

        # Compute routing logits
        routing_logits = self.routing_head(x)

        # Add dynamic bias
        routing_logits = routing_logits + self.dynamic_bias

        # Optional noise
        if self.training and self.noise_std > 0.0:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * self.noise_std

        # Softmax over experts
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Update utilization + bias
        if self.training:
            batch_util = routing_weights.mean(dim=(0, 1)).detach()

            self.expert_utilization = (
                0.99 * self.expert_utilization + 0.01 * batch_util
            )

            util_error = self.target_utilization - self.expert_utilization
            self.dynamic_bias = self.dynamic_bias + self.bias_lr * util_error

        return routing_weights


class HmoeGateTopK(nn.Module):
    """Sparse Top-K routing gate with dynamic load balancing.

    Only the top-k experts receive probability mass.

    Args:
        input_dim (int): Input feature dimension.
        num_children (int): Number of experts.
        k (int): Number of active experts.
        noise_std (float): Noise std for exploration.
    """

    def __init__(self, input_dim: int, num_children: int, k: int = 1, noise_std: float = 0.1):
        super().__init__()

        self.routing_head = nn.Linear(input_dim, num_children)
        self.k = min(k, num_children)
        self.noise_std = noise_std

        # Load balancing buffers
        self.register_buffer('expert_utilization', torch.ones(num_children) / num_children)
        self.register_buffer('dynamic_bias', torch.zeros(num_children))
        self.target_utilization = 1.0 / num_children
        self.bias_lr = 0.01

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Compute sparse routing weights.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Sparse routing weights [B, T, num_children].
        """
        raw_data = payload_tensor.to_tensor()

        # Compute logits
        routing_logits = self.routing_head(raw_data)

        # Apply dynamic bias BEFORE Top-K selection
        routing_logits = routing_logits + self.dynamic_bias

        # Add exploration noise
        if self.training and self.noise_std > 0.0:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * self.noise_std

        # Select top-k experts per token
        topk_logits, topk_indices = torch.topk(routing_logits, self.k, dim=-1)

        # Create mask filled with -inf (suppresses non-topk)
        mask = torch.full_like(routing_logits, float('-inf'))

        # Scatter top-k logits into mask
        mask.scatter_(-1, topk_indices, topk_logits)

        # Softmax over masked logits
        routing_weights = F.softmax(mask, dim=-1)

        # Update utilization based on actual routed traffic
        if self.training:
            batch_util = routing_weights.mean(dim=(0, 1)).detach()

            self.expert_utilization = (
                0.99 * self.expert_utilization + 0.01 * batch_util
            )

            util_error = self.target_utilization - self.expert_utilization
            self.dynamic_bias = self.dynamic_bias + self.bias_lr * util_error

        return routing_weights


class HmoeGateGRU(nn.Module):
    """Stateful routing gate using GRU.

    Captures temporal dependencies before computing routing decisions.

    Args:
        input_dim (int): Input feature dimension.
        num_children (int): Number of experts.
        hidden_dim (int): GRU hidden size.
        num_layers (int): Number of GRU layers.
        dropout_p (float): Dropout probability.
        noise_std (float): Noise std for exploration.
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
        super().__init__()

        self.noise_std = noise_std

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Projection to expert logits
        self.routing_head = nn.Linear(hidden_dim, num_children)

        # Load balancing buffers
        self.register_buffer('expert_utilization', torch.ones(num_children) / num_children)
        self.register_buffer('dynamic_bias', torch.zeros(num_children))
        self.target_utilization = 1.0 / num_children
        self.bias_lr = 0.01

    def forward(self, payload_tensor: HmoeTensor) -> torch.Tensor:
        """Compute routing weights using GRU features.

        Args:
            payload_tensor (HmoeTensor): Input tensor wrapper.

        Returns:
            torch.Tensor: Routing weights [B, T, num_children].
        """
        raw_data = payload_tensor.to_tensor()

        # Process sequence through GRU
        out, _ = self.gru(raw_data)

        # Compute routing logits
        routing_logits = self.routing_head(out)

        # Apply dynamic bias
        routing_logits = routing_logits + self.dynamic_bias

        # Optional exploration noise
        if self.training and self.noise_std > 0.0:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * self.noise_std

        # Convert to probabilities
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Update utilization + bias
        if self.training:
            batch_util = routing_weights.mean(dim=(0, 1)).detach()

            self.expert_utilization = (
                0.99 * self.expert_utilization + 0.01 * batch_util
            )

            util_error = self.target_utilization - self.expert_utilization
            self.dynamic_bias = self.dynamic_bias + self.bias_lr * util_error

        return routing_weights