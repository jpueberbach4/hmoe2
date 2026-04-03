import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass
from hmoe2.schema import HmoeTask
from hmoe2.tensor import HmoeTensor, HmoeOutput


@dataclass(frozen=True)
class HmoeLossResult:
    """Container for loss computation results.

    This data transfer object (DTO) encapsulates the total differentiable
    loss used for backpropagation, along with per-task scalar metrics for logging.

    Attributes:
        total_loss (torch.Tensor): Combined loss used for optimization.
        task_metrics (Dict[str, float]): Dictionary of task-specific loss values.
    """
    total_loss: torch.Tensor
    task_metrics: Dict[str, float]


class HmoeLossEngine(nn.Module):
    """Global loss computation engine for HMoE models.

    This module computes task-specific losses using a modified focal loss
    formulation inspired by CenterNet for continuous Gaussian heatmap targets.
    It also incorporates an optional routing penalty.

    Attributes:
        tasks (List[HmoeTask]): List of task configurations.
        routing_penalty_weight (float): Weight applied to routing loss penalty.
    """

    def __init__(self, tasks: List[HmoeTask], routing_penalty_weight: float = 0.05):
        """Initializes the HmoeLossEngine.

        Args:
            tasks (List[HmoeTask]): List of task configurations.
            routing_penalty_weight (float): Scaling factor for routing penalty.
        """
        super().__init__()

        # Store task configurations
        self.tasks = tasks

        # Store routing penalty weight
        self.routing_penalty_weight = routing_penalty_weight

    def forward(self, predictions: HmoeOutput, master_tensor: HmoeTensor) -> HmoeLossResult:
        """Computes the total loss and task-specific metrics.

        Args:
            predictions (HmoeOutput): Model predictions containing task logits and routing loss.
            master_tensor (HmoeTensor): Ground truth tensor containing all labels.

        Returns:
            HmoeLossResult: Structured result containing total loss and metrics.

        Raises:
            RuntimeError: If no valid task predictions are available.
        """
        # Ensure that predictions contain task outputs
        if not predictions.task_logits:
            raise RuntimeError(
                "CRITICAL ERROR: The forward pass returned zero task predictions. "
                "The dynamic task heads were not linked correctly."
            )

        # Determine device from prediction tensors
        device = next(iter(predictions.task_logits.values())).to_tensor().device

        # Initialize metric logging and list of active losses
        metrics_log: Dict[str, float] = {}
        active_losses = []

        # Iterate over all configured tasks
        for task in self.tasks:
            # Skip disabled tasks
            if not getattr(task, 'enabled', True):
                continue

            # Skip tasks without valid label targets or predictions
            if task.label_target is None or task.name not in predictions.task_logits:
                continue

            # Extract logits tensor for the current task
            logits = predictions.task_logits[task.name].to_tensor()

            # Extract corresponding ground truth labels as a DTO subset
            label_dto = master_tensor.get_subset([task.label_target])

            # Convert label DTO into raw tensor
            targets = label_dto.to_tensor()

            # Ensure target values are positive and within probability bounds
            targets = torch.abs(targets)
            targets = torch.clamp(targets, min=0.0, max=1.0)

            # Extract shape information
            batch_size, seq_len, num_classes = logits.shape

            # Flatten logits and targets for vectorized computation
            flat_logits = logits.view(-1, num_classes)
            y = targets.view(-1)

            # Compute class probabilities using softmax
            probs = F.softmax(flat_logits, dim=-1)

            # Extract probability of positive class (index 1)
            p = probs[:, 1]

            # Define focal loss hyperparameters
            alpha = 0.0
            beta = 1.0

            # Create masks to separate peak targets from non-peak regions
            pos_mask = (y >= 0.999).float()
            neg_mask = (y < 0.999).float()

            # Compute loss for positive (peak) locations
            pos_loss = -torch.log(p + 1e-8) * torch.pow(1.0 - p, alpha) * pos_mask

            # Compute loss for negative (non-peak) locations with slope weighting
            neg_loss = (
                -torch.log(1.0 - p + 1e-8)
                * torch.pow(p, alpha)
                * torch.pow(1.0 - y, beta)
                * neg_mask
            )

            # Combine positive and negative losses with task-specific weighting
            raw_loss = (pos_loss * task.pos_weight) + neg_loss

            # Count number of positive peaks and prevent division by zero
            num_pos = pos_mask.sum().clamp(min=1.0)

            # Normalize loss by number of peaks instead of total elements
            raw_loss = raw_loss.sum() / num_pos

            # Apply task-specific loss weight
            weighted_loss = raw_loss * task.loss_weight

            # Store active loss and logging metric
            active_losses.append(weighted_loss)
            metrics_log[task.name] = weighted_loss.item()

        # Ensure at least one valid task contributed to the loss
        if not active_losses:
            raise RuntimeError("Loss calculation failed: No valid tasks were processed.")

        # Aggregate all task losses into a single scalar
        task_loss_total = torch.stack(active_losses).sum()

        # Compute routing penalty scaled by configured weight
        final_routing_loss = (
            predictions.routing_loss.to_tensor().to(device)
            * self.routing_penalty_weight
        )

        # Combine task loss and routing penalty
        total_combined_loss = task_loss_total + final_routing_loss

        # Log routing penalty for monitoring
        metrics_log['routing_penalty'] = final_routing_loss.item()

        # Return structured loss result
        return HmoeLossResult(
            total_loss=total_combined_loss,
            task_metrics=metrics_log
        )