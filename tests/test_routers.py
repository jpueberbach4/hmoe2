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
def test_forward_routing_loss_aggregation(mock_features, router_with_children, dummy_payload):
    """
    Validates that the router correctly sums routing losses from its children.
    Load balancing is now handled inside the gates via dynamic EMA bias, 
    so the router no longer adds its own explicit mathematical penalty here.
    """
    mock_features.return_value = ["f1"]
    router, child1, child2 = router_with_children
    mock_gate = MagicMock()
    
    # Fake weights (routing loss computation is independent of this now)
    mock_gate.return_value = torch.zeros(2, 3, 2)
    router.gate = mock_gate
    
    out = router(dummy_payload)
    
    # Total routing loss = 0.0 (Router Gate Loss) + 0.5 (Child 1) + 0.5 (Child 2) = 1.0
    assert out.routing_loss.tensor.item() == 1.0

@patch('hmoe2.routers.HmoeRouter.subtree_features', new_callable=PropertyMock)
def test_forward_uses_local_features(mock_features, router_with_children):
    """
    Validates that a router uses its explicit local features for the gating 
    decision instead of the entire subtree if they are defined.
    """
    mock_features.return_value = ["f_child1", "f_child2"]
    router, _, _ = router_with_children
    
    # Define a local router feature
    local_feat = MagicMock()
    local_feat.name = "local_router_feat"
    router.features = [local_feat] 
    
    # Mock payload
    mock_payload = MagicMock()
    mock_payload.get_subset.return_value = mock_payload
    mock_payload.to_tensor.return_value = torch.ones(2, 3, 4)
    
    router.gate = MagicMock()
    router.gate.return_value = torch.ones(2, 3, 2)
    
    router(mock_payload)
    
    # The router should have called get_subset explicitly with its LOCAL features for the gate
    mock_payload.get_subset.assert_any_call(router.features)

