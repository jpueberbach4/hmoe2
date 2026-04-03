import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Any, Dict

from hmoe2.schema import (
    HmoeTask, HmoeNodeType
)
from hmoe2.tensor import (
    HmoeTensor, HmoeInput, HmoeOutput
)
from hmoe2.nodes import HmoeNode
from hmoe2.gates import (
    HmoeGate, HmoeGateTCN, HmoeGateTopK, HmoeGateGRU
)


@dataclass(eq=False)
class HmoeRouter(HmoeNode):
    """Routing node for hierarchical Mixture-of-Experts.

    This node dynamically routes input data to multiple child branches using
    a learned gating mechanism. It aggregates outputs from child nodes based
    on routing weights and applies a load-balancing penalty.

    Attributes:
        branches (nn.ModuleList): List of child nodes (experts or routers).
        gate (nn.Module): Routing gate responsible for computing weights.
        config (Dict[str, Any]): Configuration dictionary for gate behavior.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initializes the HmoeRouter.

        Args:
            name (str): Unique name of the router node.
            config (Dict[str, Any], optional): Configuration for gate selection and parameters.
        """
        nn.Module.__init__(self)
        super().__init__(type=HmoeNodeType.ROUTER, name=name)

        # Initialize container for child branches
        self.branches = nn.ModuleList()

        # Placeholder for routing gate (constructed later)
        self.gate = None

        # Store configuration dictionary (fallback to empty dict if None)
        self.config = config or {}

    def build_gate(self) -> None:
        """Constructs the routing gate based on configuration.

        This method determines the appropriate gate type and initializes
        it using the current subtree feature dimensionality and number of children.
        """
        # Determine input dimensionality from subtree features
        input_dim = len(self.subtree_features)

        # Determine number of child branches
        num_children = len(self.branches)

        # Extract gate configuration parameters
        gate_type = self.config.get('gate_type', 'TCN').upper()
        noise_std = self.config.get('noise_std', 0.1)

        # Instantiate appropriate gate type
        if gate_type == 'LINEAR':
            self.gate = HmoeGate(input_dim=input_dim, num_children=num_children)
            self.gate.noise_std = noise_std

        elif gate_type == 'TOPK':
            # Retrieve top-k parameter for sparse routing
            k_val = self.config.get('top_k', 1)
            self.gate = HmoeGateTopK(input_dim, num_children, k=k_val, noise_std=noise_std)

        elif gate_type == 'GRU':
            # Retrieve hidden dimension for GRU-based gate
            hidden_dim = self.config.get('hidden_dim', 32)
            self.gate = HmoeGateGRU(input_dim, num_children, hidden_dim=hidden_dim, noise_std=noise_std)

        else:
            # Default to TCN-based gate
            self.gate = HmoeGateTCN(
                input_dim=input_dim,
                num_children=num_children,
                noise_std=noise_std
            )

    def _serialize_node(self) -> Dict[str, Any]:
        """Serializes the router node configuration.

        Returns:
            Dict[str, Any]: Dictionary representation of the router.
        """
        # Construct dictionary including gate configuration and serialized children
        return {
            'name': self.name,
            'type': self.type.name,
            'gate_type': self.config.get('gate_type', 'TCN'),
            'noise_std': self.config.get('noise_std', 0.1),
            'children': [child._serialize_node() for child in self.branches]
        }

    def _gather_tasks(self, task_dict: Dict[str, HmoeTask]) -> None:
        """Aggregates tasks from all child branches.

        Args:
            task_dict (Dict[str, HmoeTask]): Dictionary to populate with tasks.
        """
        # Recursively collect tasks from each child node
        for child in self.branches:
            child._gather_tasks(task_dict)

    def link_tasks(self, global_tasks: List[HmoeTask]) -> None:
        """Propagates task linking to all child nodes.

        Args:
            global_tasks (List[HmoeTask]): List of globally defined tasks.
        """
        # Delegate task linking to children that support it
        for child in self.branches:
            if hasattr(child, 'link_tasks'):
                child.link_tasks(global_tasks)

    def forward(self, payload: HmoeInput) -> HmoeOutput:
        """Routes input through child branches and aggregates outputs.

        Args:
            payload (HmoeInput): Input payload containing feature data.

        Returns:
            HmoeOutput: Aggregated output with routing loss.
        """
        # Determine features required by all child branches
        branch_features = self.subtree_features

        # Narrow payload to only required features
        narrowed_payload = payload.get_subset(branch_features)

        # Compute routing weights using the gate
        gate_weights = self.gate(narrowed_payload)

        # Collect outputs from each child branch
        child_outputs: List[HmoeOutput] = []
        for child in self.branches:
            child_outputs.append(child(narrowed_payload))

        # Initialize structure for aggregated task logits
        pooled_task_logits: Dict[str, HmoeTensor] = {}

        # Determine all unique task names across children
        all_task_names = set()
        for out in child_outputs:
            all_task_names.update(out.task_logits.keys())

        # Aggregate outputs for each task
        for task_name in all_task_names:
            weighted_tensors = []
            ref_indices = None

            # Iterate over child outputs and apply routing weights
            for child_idx, child_out in enumerate(child_outputs):
                if task_name in child_out.task_logits:
                    tensor = child_out.task_logits[task_name].to_tensor()

                    # Capture indices from first occurrence
                    if ref_indices is None:
                        ref_indices = child_out.task_logits[task_name].get_indices()

                    # Extract routing weight for this child and expand dimensions
                    weight = gate_weights[:, :, child_idx].unsqueeze(-1)

                    # Apply weight to child output tensor
                    weighted_tensors.append(tensor * weight)

            # Sum weighted tensors across children
            gated_data = torch.stack(weighted_tensors).sum(dim=0)

            # Store aggregated result in HmoeTensor format
            pooled_task_logits[task_name] = HmoeTensor(
                tensor=gated_data,
                indices=ref_indices
            )

        # Compute load-balancing penalty for routing distribution
        num_branches = len(self.branches)

        if num_branches > 1:
            # Compute average routing probability per branch
            mean_routing_probs = gate_weights.mean(dim=(0, 1))

            # Encourage uniform distribution across branches
            balance_loss = (num_branches * torch.sum(mean_routing_probs ** 2)) - 1.0
        else:
            # No penalty when only one branch exists
            balance_loss = torch.tensor(0.0, device=gate_weights.device)

        # Aggregate routing losses from child nodes
        child_routing_loss_raw = torch.tensor(0.0, device=gate_weights.device)

        for out in child_outputs:
            # Only include routing loss if present
            if getattr(out, 'routing_loss', None) is not None:
                child_routing_loss_raw += out.routing_loss.to_tensor().to(gate_weights.device)

        # Combine local balance loss with child routing losses
        total_routing_loss_raw = balance_loss + child_routing_loss_raw

        # Return final output with aggregated logits and routing loss
        return HmoeOutput(
            task_logits=pooled_task_logits,
            routing_loss=HmoeTensor(tensor=total_routing_loss_raw, indices=[])
        )