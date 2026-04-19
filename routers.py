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
    """Hierarchical routing node for Mixture-of-Experts.

    This node:
    - Computes routing weights using a configurable gating mechanism
    - Dispatches inputs to child nodes (experts or sub-routers)
    - Aggregates outputs using weighted combination
    - Propagates routing loss upward

    Args:
        name (str): Node identifier.
        config (Dict[str, Any], optional): Router configuration.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        nn.Module.__init__(self)
        super().__init__(type=HmoeNodeType.ROUTER, name=name)

        # Child nodes (experts or sub-routers)
        self.branches = nn.ModuleList()

        # Routing gate (set later via build_gate)
        self.gate = None

        # Configuration dictionary (gate type, noise, etc.)
        self.config = config or {}

    def build_gate(self) -> None:
        """Initialize routing gate based on configuration.

        Selects input dimension and gate type dynamically.
        """
        # Determine feature dimension used for routing
        if len(self.features) > 0:
            input_dim = len(self.features)  # router-specific features
        else:
            input_dim = len(self.subtree_features)  # fallback to full subtree

        num_children = len(self.branches)

        # Configuration options
        gate_type = self.config.get('gate_type', 'TCN').upper()
        noise_std = self.config.get('noise_std', 0.1)

        # Pass-through routing (uniform weights)
        if gate_type == 'PASS_THROUGH':
            self.gate = None
            return

        # Select gate implementation
        if gate_type == 'LINEAR':
            self.gate = HmoeGate(input_dim=input_dim, num_children=num_children)
            self.gate.noise_std = noise_std

        elif gate_type == 'TOPK':
            k_val = self.config.get('top_k', 1)
            self.gate = HmoeGateTopK(
                input_dim, num_children,
                k=k_val,
                noise_std=noise_std
            )

        elif gate_type == 'GRU':
            hidden_dim = self.config.get('hidden_dim', 32)
            self.gate = HmoeGateGRU(
                input_dim,
                num_children,
                hidden_dim=hidden_dim,
                noise_std=noise_std
            )

        else:
            # Default: temporal convolutional gate
            self.gate = HmoeGateTCN(
                input_dim=input_dim,
                num_children=num_children,
                noise_std=noise_std
            )

    def _serialize_node(self) -> Dict[str, Any]:
        """Serialize router structure recursively.

        Returns:
            Dict[str, Any]: JSON-like representation of the node.
        """
        return {
            'name': self.name,
            'type': self.type.name,
            'gate_type': self.config.get('gate_type', 'TCN'),
            'noise_std': self.config.get('noise_std', 0.1),
            'features': [f.name for f in self.features] if self.features else [],
            'children': [child._serialize_node() for child in self.branches]
        }

    def _gather_tasks(self, task_dict: Dict[str, HmoeTask]) -> None:
        """Collect tasks from all children recursively."""
        for child in self.branches:
            child._gather_tasks(task_dict)

    def link_tasks(self, global_tasks: List[HmoeTask]) -> None:
        """Propagate global task definitions to children."""
        for child in self.branches:
            if hasattr(child, 'link_tasks'):
                child.link_tasks(global_tasks)

    def forward(self, payload: HmoeInput) -> HmoeOutput:
        """Execute routing and aggregation.

        Args:
            payload (HmoeInput): Input data wrapper.

        Returns:
            HmoeOutput: Aggregated outputs and routing loss.
        """
        # Extract features used by children
        child_features = self.subtree_features

        # Payload passed to children
        child_payload = payload.get_subset(child_features)

        # Payload used for routing decision
        if len(self.features) > 0:
            router_payload = payload.get_subset(self.features)
        else:
            router_payload = child_payload

        # Compute routing weights [B, T, num_children]
        if self.gate is None:
            # Uniform routing (pass-through)
            raw_t = router_payload.to_tensor()

            gate_weights = torch.ones(
                raw_t.size(0),  # batch
                raw_t.size(1),  # sequence length
                len(self.branches),  # number of experts
                device=raw_t.device
            )
        else:
            gate_weights = self.gate(router_payload)

        # Execute all child nodes
        child_outputs: List[HmoeOutput] = []
        for child in self.branches:
            child_outputs.append(child(child_payload))

        # Collect all task names across children
        pooled_task_logits: Dict[str, HmoeTensor] = {}
        all_task_names = set()

        for out in child_outputs:
            all_task_names.update(out.task_logits.keys())

        # Aggregate outputs per task
        for task_name in all_task_names:
            weighted_tensors = []
            ref_indices = None  # used to preserve metadata

            # Combine contributions from each child
            for child_idx, child_out in enumerate(child_outputs):
                if task_name in child_out.task_logits:
                    tensor = child_out.task_logits[task_name].to_tensor()

                    # Store indices once (assumed consistent across children)
                    if ref_indices is None:
                        ref_indices = child_out.task_logits[task_name].get_indices()

                    # Extract routing weights for this child [B, T, 1]
                    weight = gate_weights[:, :, child_idx].unsqueeze(-1)

                    # Apply weighting
                    weighted_tensors.append(tensor * weight)

            # Sum weighted contributions across children
            gated_data = torch.stack(weighted_tensors).sum(dim=0)

            # Wrap back into HmoeTensor
            pooled_task_logits[task_name] = HmoeTensor(
                tensor=gated_data,
                indices=ref_indices
            )

        # No explicit auxiliary routing loss at this level
        # (load balancing handled inside gates via dynamic bias)
        total_gate_loss = torch.tensor(
            0.0,
            device=router_payload.to_tensor().device
        )

        # Accumulate routing loss from children
        child_routing_loss_raw = torch.tensor(
            0.0,
            device=router_payload.to_tensor().device
        )

        for out in child_outputs:
            if getattr(out, 'routing_loss', None) is not None:
                child_routing_loss_raw += (
                    out.routing_loss
                    .to_tensor()
                    .to(router_payload.to_tensor().device)
                )

        # Total routing loss passed upward
        total_routing_loss_raw = total_gate_loss + child_routing_loss_raw

        return HmoeOutput(
            task_logits=pooled_task_logits,
            routing_loss=HmoeTensor(
                tensor=total_routing_loss_raw,
                indices=[]
            )
        )