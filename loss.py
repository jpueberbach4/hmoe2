import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass

from hmoe2.schema import HmoeTask
from hmoe2.tensor import HmoeTensor, HmoeOutput


@dataclass(frozen=True)
class HmoeLossResult:
    """Container for loss outputs.

    Attributes:
        total_loss (torch.Tensor): Combined loss (task + routing penalty).
        task_metrics (Dict[str, float]): Per-task loss values for logging.
    """
    total_loss: torch.Tensor
    task_metrics: Dict[str, float]


class HmoeLossEngine(nn.Module):
    """Multi-task loss computation engine for HMoE.

    Combines:
    - Per-task classification loss (CrossEntropy + Focal modulation)
    - Routing penalty (encourages balanced expert usage)

    Args:
        tasks (List[HmoeTask]): List of task definitions.
        routing_penalty_weight (float): Weight applied to routing loss.
    """

    def __init__(self, tasks: List[HmoeTask], routing_penalty_weight: float = 0.05):
        super().__init__()

        # Task configuration (defines labels, weights, enable flags)
        self.tasks = tasks

        # Scaling factor for routing regularization term
        self.routing_penalty_weight = routing_penalty_weight

    def forward(
        self,
        predictions: HmoeOutput,
        master_tensor: HmoeTensor
    ) -> HmoeLossResult:
        """Compute total loss across all tasks.

        Args:
            predictions (HmoeOutput): Model outputs (logits + routing loss).
            master_tensor (HmoeTensor): Input tensor containing ground truth labels.

        Returns:
            HmoeLossResult: Aggregated loss and per-task metrics.

        Raises:
            RuntimeError: If no predictions or no valid tasks are found.
        """
        # Ensure model produced at least one task output
        if not predictions.task_logits:
            raise RuntimeError("CRITICAL ERROR: Zero task predictions returned.")

        # Determine device from any prediction tensor
        device = next(iter(predictions.task_logits.values())).to_tensor().device

        # Dictionary for logging per-task losses
        metrics_log: Dict[str, float] = {}

        # Accumulates active task losses
        active_losses = []

        # Iterate over configured tasks
        for task in self.tasks:
            # Skip disabled tasks or tasks without valid labels/predictions
            if (
                not getattr(task, 'enabled', True)
                or task.label_target is None
                or task.name not in predictions.task_logits
            ):
                continue

            # Extract logits: [B, T, C]
            logits = predictions.task_logits[task.name].to_tensor()

            # Extract targets and ensure correct dtype for CE loss
            targets = (
                master_tensor
                .get_subset([task.label_target])
                .to_tensor()
                .squeeze(-1)
                .long()
            )

            # Shapes
            batch_size, seq_len, num_classes = logits.shape

            # Flatten temporal dimension for CE computation
            flat_logits = logits.view(-1, num_classes)  # [(B*T), C]
            y = targets.view(-1)                        # [(B*T)]

            # Standard cross-entropy loss per element (no reduction)
            ce_loss = F.cross_entropy(flat_logits, y, reduction='none')

            # Convert CE -> focal loss
            # pt = probability assigned to correct class
            pt = torch.exp(-ce_loss)

            # Gamma controls how strongly easy examples are down-weighted
            gamma = 2.0

            # Focal scaling reduces gradient for confident predictions
            focal_loss = ((1 - pt) ** gamma) * ce_loss

            # Mean over all tokens, scaled by task-specific weight
            task_weight = getattr(task, 'loss_weight', 1.0)
            weighted_loss = focal_loss.mean() * task_weight

            # Store loss for aggregation
            active_losses.append(weighted_loss)

            # Log scalar value for monitoring
            metrics_log[task.name] = weighted_loss.item()

        # Ensure at least one task contributed to the loss
        if not active_losses:
            raise RuntimeError("Loss calculation failed: No valid tasks processed.")

        # Sum all task losses
        task_loss_total = torch.stack(active_losses).sum()

        # Routing penalty encourages balanced expert usage
        routing_loss = predictions.routing_loss.to_tensor().to(device)
        final_routing_loss = routing_loss * self.routing_penalty_weight

        # Combine task + routing losses
        total_combined_loss = task_loss_total + final_routing_loss

        # Log routing penalty separately
        metrics_log['routing_penalty'] = final_routing_loss.item()

        return HmoeLossResult(
            total_loss=total_combined_loss,
            task_metrics=metrics_log
        )