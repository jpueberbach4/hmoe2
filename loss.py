import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass
from hmoe2.schema import HmoeTask
from hmoe2.tensor import HmoeTensor, HmoeOutput


@dataclass(frozen=True)
class HmoeLossResult:
    """Container for loss computation outputs.

    Attributes:
        total_loss (torch.Tensor):
            Final scalar loss combining task losses and routing penalty.
        task_metrics (Dict[str, float]):
            Per-task loss values (detached scalars for logging/monitoring).
    """
    total_loss: torch.Tensor
    task_metrics: Dict[str, float]


class HmoeLossEngine(nn.Module):
    """Loss computation engine for Hierarchical Mixture-of-Experts (HMoE).

    This module aggregates task-specific classification losses and combines
    them with a routing penalty term. Each task contributes independently
    based on its configuration (e.g., weight, positive class scaling).

    Attributes:
        tasks (List[HmoeTask]):
            List of task definitions containing metadata such as name,
            label target, and weighting parameters.
        routing_penalty_weight (float):
            Scaling factor applied to the routing loss component.
    """

    def __init__(self, tasks: List[HmoeTask], routing_penalty_weight: float = 0.05):
        """Initializes the loss engine.

        Args:
            tasks (List[HmoeTask]):
                List of task configurations.
            routing_penalty_weight (float, optional):
                Weight applied to the routing penalty term. Defaults to 0.05.
        """
        super().__init__()
        self.tasks = tasks
        self.routing_penalty_weight = routing_penalty_weight

    def forward(self, predictions: HmoeOutput, master_tensor: HmoeTensor) -> HmoeLossResult:
        """Computes the combined loss across all active tasks.

        For each task:
        - Extract logits and targets
        - Apply binary-safe preprocessing
        - Compute log-softmax probabilities
        - Apply masked positive/negative loss
        - Normalize and weight the loss

        Finally, adds the routing penalty to produce the total loss.

        Args:
            predictions (HmoeOutput):
                Model output containing task logits and routing loss.
            master_tensor (HmoeTensor):
                Input tensor containing ground-truth labels.

        Returns:
            HmoeLossResult:
                Object containing total loss and per-task metrics.

        Raises:
            RuntimeError:
                If no task predictions are available or no valid tasks are processed.
        """
        # Ensure model returned at least one task prediction
        if not predictions.task_logits:
            raise RuntimeError("CRITICAL ERROR: Zero task predictions returned.")

        # Infer device from predictions
        device = next(iter(predictions.task_logits.values())).to_tensor().device

        # Store per-task scalar metrics for logging
        metrics_log: Dict[str, float] = {}

        # Accumulate valid task losses
        active_losses = []

        for task in self.tasks:
            # Skip disabled or improperly configured tasks
            if not getattr(task, 'enabled', True) or task.label_target is None or task.name not in predictions.task_logits:
                continue

            # Extract logits and corresponding targets
            logits = predictions.task_logits[task.name].to_tensor()
            targets = master_tensor.get_subset([task.label_target]).to_tensor()

            # Ensure targets are strictly non-negative and within [0, 1]
            targets = torch.abs(targets)
            y_clamped = torch.clamp(targets, min=0.0, max=1.0)

            batch_size, seq_len, num_classes = logits.shape

            # Flatten tensors for vectorized loss computation
            flat_logits = logits.view(-1, num_classes)
            y = y_clamped.view(-1)

            # Compute log-probabilities (numerically stable)
            log_probs = F.log_softmax(flat_logits, dim=-1)
            log_p_pos = log_probs[:, 1]
            log_p_neg = log_probs[:, 0]

            # REPLACED: Use continuous soft-targets instead of binary masks
            # If y = 0.8, it heavily penalizes missing the pos class (80%), 
            # and slightly penalizes missing the neg class (20%).
            pos_loss = -log_p_pos * y
            neg_loss = -log_p_neg * (1.0 - y)

            # Combine losses with task-specific positive weighting
            raw_loss = (pos_loss * task.pos_weight) + neg_loss

            # Normalize by the sum of target values (to handle soft totals)
            num_pos = y.sum().clamp(min=1.0)
            raw_loss = raw_loss.sum() / num_pos

            # Apply task-level weighting
            weighted_loss = raw_loss * task.loss_weight

            active_losses.append(weighted_loss)
            metrics_log[task.name] = weighted_loss.item()

        # Ensure at least one task contributed to the loss
        if not active_losses:
            raise RuntimeError("Loss calculation failed: No valid tasks processed.")

        # Aggregate all task losses
        task_loss_total = torch.stack(active_losses).sum()

        # Add routing penalty (scaled)
        final_routing_loss = predictions.routing_loss.to_tensor().to(device) * self.routing_penalty_weight
        total_combined_loss = task_loss_total + final_routing_loss

        # Log routing penalty separately
        metrics_log['routing_penalty'] = final_routing_loss.item()

        return HmoeLossResult(
            total_loss=total_combined_loss,
            task_metrics=metrics_log
        )