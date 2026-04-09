import pytest
import torch
import torch.nn as nn
from typing import Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

from hmoe2.routers import HmoeRouter
from hmoe2.tensor import HmoeTensor, HmoeInput, HmoeOutput
from hmoe2.schema import HmoeTask

# ==========================================
# MOCKS AND FIXTURES
# ==========================================

class DummyChild(nn.Module):
    """A PyTorch-native mock for child branches (experts or other routers)."""
    def __init__(self, name, output_tensor, routing_loss_val=0.0):
        super().__init__()
        self.node_name = name
        self.output_tensor = output_tensor
        self.routing_loss_val = routing_loss_val
        self.gathered_tasks = []
        self.linked_tasks = []

    def _serialize_node(self):
        return {"name": self.node_name, "type": "DUMMY_CHILD"}

    def _gather_tasks(self, task_dict):
        task_dict[f"{self.node_name}_task"] = MagicMock(name="HmoeTask")

    def link_tasks(self, global_tasks):
        self.linked_tasks = global_tasks

    def forward(self, payload):
        # Create a mock HmoeOutput that mimics a child's prediction
        mock_task_tensor = MagicMock()
        mock_task_tensor.to_tensor.return_value = self.output_tensor
        mock_task_tensor.get_indices.return_value = ["class_0", "class_1"]

        mock_routing_tensor = MagicMock()
        mock_routing_tensor.to_tensor.return_value = torch.tensor(self.routing_loss_val)

        return MagicMock(
            task_logits={"shared_task": mock_task_tensor},
            routing_loss=mock_routing_tensor
        )


@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 3

@pytest.fixture
def dummy_payload():
    """Mocks the HmoeInput to return a generic tensor when queried."""
    mock_payload = MagicMock()
    mock_payload.get_subset.return_value = mock_payload
    mock_payload.to_tensor.return_value = torch.randn(2, 3, 4) # Batch=2, Seq=3, Features=4
    return mock_payload


@pytest.fixture
def router_with_children():
    """Yields a router pre-populated with two dummy child branches."""
    router = HmoeRouter(name="test_router")
    
    # Child 1 outputs all 1s
    child1 = DummyChild("child1", torch.ones(2, 3, 2), routing_loss_val=0.5)
    # Child 2 outputs all 2s
    child2 = DummyChild("child2", torch.ones(2, 3, 2) * 2.0, routing_loss_val=0.5)
    
    router.branches.extend([child1, child2])
    return router, child1, child2


# ==========================================
# INITIALIZATION & GATE BUILDING TESTS
# ==========================================

def test_router_initialization():
    """Validates default initialization attributes."""
    router = HmoeRouter(name="root")
    
    assert router.name == "root"
    assert router.type.name == "ROUTER"
    assert len(router.branches) == 0
    assert router.gate is None
    assert router.config == {}

@patch('hmoe2.routers.HmoeGateTCN')
@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_build_gate_default_tcn(mock_features, MockGateTCN):
    """Validates the default fallback gate is a TCN."""
    mock_features.return_value = ["f1", "f2"] # 2 features
    
    router = HmoeRouter(name="root", config={})
    router.branches.extend([nn.Module(), nn.Module(), nn.Module()]) # 3 children
    
    router.build_gate()
    
    MockGateTCN.assert_called_once_with(input_dim=2, num_children=3, noise_std=0.1)
    assert router.gate == MockGateTCN.return_value

@patch('hmoe2.routers.HmoeGateTopK')
@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_build_gate_topk(mock_features, MockGateTopK):
    """Validates parsing of TOPK configuration parameters."""
    mock_features.return_value = ["f1"]
    
    config = {'gate_type': 'TOPK', 'top_k': 2, 'noise_std': 0.2}
    router = HmoeRouter(name="root", config=config)
    router.branches.extend([nn.Module(), nn.Module(), nn.Module()])
    
    router.build_gate()
    
    MockGateTopK.assert_called_once_with(1, 3, k=2, noise_std=0.2)
    assert router.gate == MockGateTopK.return_value

@patch('hmoe2.routers.HmoeGateGRU')
@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_build_gate_gru(mock_features, MockGateGRU):
    """Validates parsing of GRU configuration parameters."""
    mock_features.return_value = ["f1"]
    
    config = {'gate_type': 'GRU', 'hidden_dim': 64}
    router = HmoeRouter(name="root", config=config)
    router.branches.extend([nn.Module(), nn.Module()])
    
    router.build_gate()
    
    MockGateGRU.assert_called_once_with(1, 2, hidden_dim=64, noise_std=0.1)
    assert router.gate == MockGateGRU.return_value

@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_build_gate_pass_through(mock_features):
    """Validates the PASS_THROUGH strategy entirely bypasses gate construction."""
    config = {'gate_type': 'PASS_THROUGH'}
    router = HmoeRouter(name="root", config=config)
    
    router.build_gate()
    
    assert router.gate is None


# ==========================================
# SERIALIZATION & DELEGATION TESTS
# ==========================================

def test_serialize_node(router_with_children):
    """Validates that router serialization recursively captures children."""
    router, child1, child2 = router_with_children
    router.config = {'gate_type': 'TOPK', 'noise_std': 0.5}
    
    serialized = router._serialize_node()
    
    assert serialized['name'] == "test_router"
    assert serialized['type'] == "ROUTER"
    assert serialized['gate_type'] == "TOPK"
    assert serialized['noise_std'] == 0.5
    
    # Children should be cleanly serialized
    assert len(serialized['children']) == 2
    assert serialized['children'][0] == {"name": "child1", "type": "DUMMY_CHILD"}
    assert serialized['children'][1] == {"name": "child2", "type": "DUMMY_CHILD"}

def test_gather_tasks(router_with_children):
    """Validates that tasks are gathered from all branches."""
    router, child1, child2 = router_with_children
    task_dict = {}
    
    router._gather_tasks(task_dict)
    
    assert len(task_dict) == 2
    assert "child1_task" in task_dict
    assert "child2_task" in task_dict

def test_link_tasks(router_with_children):
    """Validates task definitions are passed down the tree."""
    router, child1, child2 = router_with_children
    global_tasks = ["mock_task_a", "mock_task_b"]
    
    router.link_tasks(global_tasks)
    
    assert child1.linked_tasks == global_tasks
    assert child2.linked_tasks == global_tasks


# ==========================================
# FORWARD PASS & LOGIT AGGREGATION TESTS
# ==========================================

@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_forward_pass_through_aggregation(mock_features, router_with_children, dummy_payload):
    """
    Validates parallel execution (PASS_THROUGH) where no gate exists.
    All children receive 100% traffic, and their outputs are simply summed.
    """
    mock_features.return_value = ["f1"]
    router, child1, child2 = router_with_children
    
    # Force pass-through routing
    router.gate = None
    
    output = router(dummy_payload)
    
    # Validate the task logits dictionary is correctly shaped
    assert "shared_task" in output.task_logits
    
    result_tensor = output.task_logits["shared_task"].tensor
    
    # Since Child 1 outputs 1s, Child 2 outputs 2s, and gate weights are 1.0 (Pass-Through),
    # The sum should equal exactly 3.0 everywhere.
    assert torch.allclose(result_tensor, torch.full((2, 3, 2), 3.0))
    
    # Load balancing penalty should be 0.0 for pass-through
    # But total routing loss = balance_loss (0) + child1_loss (0.5) + child2_loss (0.5)
    assert output.routing_loss.tensor.item() == 1.0

@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_forward_gated_aggregation(mock_features, router_with_children, dummy_payload):
    """
    Validates weighted aggregation based on gate probabilities.
    Ensures tensor scaling logic perfectly maps gate [Batch, Seq, Experts] 
    to output [Batch, Seq, Classes].
    """
    mock_features.return_value = ["f1"]
    router, child1, child2 = router_with_children
    
    # Create a dummy gate that assigns 80% probability to child1 and 20% to child2
    mock_gate = MagicMock()
    
    # Shape: [Batch=2, Seq=3, Children=2]
    # We set weight[..., 0] = 0.8 and weight[..., 1] = 0.2
    dummy_gate_weights = torch.zeros(2, 3, 2)
    dummy_gate_weights[:, :, 0] = 0.8
    dummy_gate_weights[:, :, 1] = 0.2
    
    mock_gate.return_value = dummy_gate_weights
    router.gate = mock_gate
    
    output = router(dummy_payload)
    result_tensor = output.task_logits["shared_task"].tensor
    
    # Math validation:
    # Child 1 (1.0) * 0.8 = 0.8
    # Child 2 (2.0) * 0.2 = 0.4
    # Expected sum = 1.2
    assert torch.allclose(result_tensor, torch.full((2, 3, 2), 1.2))

@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_forward_load_balancing_penalty(mock_features, router_with_children, dummy_payload):
    """
    Validates the mathematical precision of the load-balancing penalty.
    It discourages the gate from collapsing all traffic to a single expert.
    """
    mock_features.return_value = ["f1"]
    router, child1, child2 = router_with_children
    mock_gate = MagicMock()
    
    # Scenario A: Perfect Collapse (Terrible balance) -> 100% to Child 1
    collapsed_weights = torch.zeros(2, 3, 2)
    collapsed_weights[:, :, 0] = 1.0 
    mock_gate.return_value = collapsed_weights
    router.gate = mock_gate
    
    out_collapsed = router(dummy_payload)
    # Math: mean_probs = [1.0, 0.0] -> sum(sq) = 1.0
    # Penalty = (2 * 1.0) - 1.0 = 1.0
    # Add child losses (0.5 + 0.5 = 1.0) -> Total = 2.0
    assert out_collapsed.routing_loss.tensor.item() == 2.0
    
    # Scenario B: Perfect Balance (Ideal) -> 50/50 split
    balanced_weights = torch.zeros(2, 3, 2)
    balanced_weights[:, :, :] = 0.5
    mock_gate.return_value = balanced_weights
    router.gate = mock_gate
    
    out_balanced = router(dummy_payload)
    # Math: mean_probs = [0.5, 0.5] -> sum(sq) = (0.25 + 0.25) = 0.5
    # Penalty = (2 * 0.5) - 1.0 = 0.0
    # Add child losses (0.5 + 0.5 = 1.0) -> Total = 1.0
    assert out_balanced.routing_loss.tensor.item() == 1.0