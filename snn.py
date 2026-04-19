import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SurrogateSpike(torch.autograd.Function):
    """Custom autograd function for spiking neurons using surrogate gradients.

    Forward pass implements a hard threshold (binary spike).
    Backward pass approximates gradients using a smooth sigmoid function.

    This allows training of non-differentiable spike operations.
    """

    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: float) -> torch.Tensor:
        """Forward spike computation.

        Args:
            mem (torch.Tensor): Membrane potential.
            threshold (float): Spike threshold.

        Returns:
            torch.Tensor: Binary spikes (0 or 1).
        """
        # Save tensors for backward pass
        ctx.save_for_backward(mem, torch.tensor(threshold))

        # Hard thresholding (non-differentiable)
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using surrogate gradient."""
        mem, threshold = ctx.saved_tensors

        # Bereken de Sigmoid benadering
        alpha = 5.0
        sig = torch.sigmoid(alpha * (mem - threshold))

        # De WARE afgeleide van de Sigmoid: f(x) * (1 - f(x))
        # We vermenigvuldigen met alpha voor de chain-rule van de binnenste functie
        surrogate_grad = alpha * sig * (1.0 - sig)

        # Propagate gradient through surrogate
        grad_input = grad_output * surrogate_grad

        return grad_input, None


class SnnBackend(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neural network backend.

    This module:
    - Converts inputs into currents via an MLP
    - Simulates membrane potential dynamics over time
    - Generates sparse spike outputs using thresholding
    - Uses surrogate gradients for training

    Key properties:
    - Strictly causal (state evolves forward in time)
    - Sparse binary activations (energy efficient)
    - Temporal memory via membrane decay

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Number of neurons.
        config (Dict): Configuration dictionary.
    """

    def __init__(self, input_dim: int, hidden_dim: int, config: Dict = {}):
        super().__init__()

        # Membrane decay factor (controls memory retention)
        self.beta = config.get('beta', 0.9)

        # Spike threshold
        self.threshold = config.get('threshold', 1.0)

        dropout_p = config.get('dropout', 0.2)

        # Input transformation into synaptic current
        # MLP allows directional/selective activation instead of linear mapping
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Output dropout applied to spike tensor
        self.out_proj = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate spiking dynamics over time.

        Args:
            x (torch.Tensor): Input tensor [B, T, D].

        Returns:
            torch.Tensor: Spike tensor [B, T, hidden_dim].
        """
        # Extract dimensions
        b, seq_len, _ = x.size()
        device = x.device

        # Convert input features to synaptic currents
        currents = self.fc(x)  # [B, T, H]

        # Initialize membrane potential (state per neuron)
        mem = torch.zeros(
            b,
            self.fc[-1].out_features,
            device=device
        )

        # Collect spikes over time
        spike_sequence = []

        # Iterate over time (strict causal processing)
        for t in range(seq_len):
            # Leaky integration:
            # previous state decays + new input current added
            mem = (self.beta * mem) + currents[:, t, :]

            # Generate spikes using threshold function
            spikes = SurrogateSpike.apply(mem, self.threshold)

            # Reset membrane where spikes occurred
            # (subtract threshold rather than zeroing for stability)
            mem = mem - (spikes * self.threshold)

            # Store spike output
            spike_sequence.append(spikes)

        # Stack time dimension back
        spikes_tensor = torch.stack(spike_sequence, dim=1)

        # Apply output dropout
        return self.out_proj(spikes_tensor)