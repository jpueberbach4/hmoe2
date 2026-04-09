import pytest
import torch
import torch.nn as nn
from typing import Dict
from unittest.mock import MagicMock, patch

from hmoe2.loss import HmoeLossEngine, HmoeLossResult
from hmoe2.schema import HmoeTask

# ==========================================
# MOCKS AND FIXTURES
# ==========================================

class MockHmoeTensor:
    def __init__(self, tensor_data: torch.Tensor):
        self.tensor_data = tensor_data
    
    def to_tensor(self):
        return self.tensor_data

    def get_subset(self, labels):
        # Simply return self for testing purposes
        return self

class MockHmoeOutput:
    def __init__(self, task_logits: Dict[str, MockHmoeTensor], routing_loss: MockHmoeTensor):
        self.task_logits = task_logits
        self.routing_loss = routing_loss

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 3

@pytest.fixture
def dummy_routing_loss():
    return torch.tensor(1.5)

@pytest.fixture
def task_a():
    task = MagicMock()
    task.name = "task_a"
    task.label_target = "target_a"
    task.pos_weight = 2.0
    task.loss_weight = 1.0
    task.enabled = True
    return task

@pytest.fixture
def task_b():
    task = MagicMock()
    task.name = "task_b"
    task.label_target = "target_b"
    task.pos_weight = 1.0
    task.loss_weight = 0.5
    task.enabled = True
    return task


# ==========================================
# TESTS
# ==========================================

def test_engine_initialization(task_a, task_b):
    """Verifies that the engine initializes and stores configuration properly."""
    tasks = [task_a, task_b]
    engine = HmoeLossEngine(tasks=tasks, routing_penalty_weight=0.1)
    
    assert engine.tasks == tasks
    assert engine.routing_penalty_weight == 0.1

def test_engine_empty_predictions_raises_error(task_a):
    """Verifies that the engine aborts if the model returns no task logits."""
    engine = HmoeLossEngine(tasks=[task_a])
    empty_output = MockHmoeOutput(task_logits={}, routing_loss=MockHmoeTensor(torch.tensor(0.0)))
    dummy_master = MockHmoeTensor(torch.tensor([1.0]))

    with pytest.raises(RuntimeError, match="CRITICAL ERROR: Zero task predictions returned."):
        engine(empty_output, dummy_master)

def test_engine_soft_target_loss_computation(task_a, batch_size, seq_len):
    """
    Validates the exact math of the continuous soft-target replacement.
    Using a 1x1 tensor to manually calculate and compare the expected loss.
    """
    engine = HmoeLossEngine(tasks=[task_a], routing_penalty_weight=0.0)
    
    # 1. Setup deterministic input logits [Batch=1, Seq=1, Classes=2 (Neg, Pos)]
    # Logits: [0.0, 0.0] -> Softmax: [0.5, 0.5] -> LogSoftmax: [ln(0.5), ln(0.5)] ≈ [-0.693, -0.693]
    logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    
    # 2. Setup a soft target of 0.8
    # y = 0.8
    targets = torch.tensor([[[0.8]]])

    predictions = MockHmoeOutput(
        task_logits={"task_a": MockHmoeTensor(logits)},
        routing_loss=MockHmoeTensor(torch.tensor(0.0))
    )
    master_tensor = MockHmoeTensor(targets)

    # 3. Calculate expected mathematically:
    # log_p_pos = -0.693147, log_p_neg = -0.693147
    # pos_loss = -(-0.693147) * 0.8 = 0.5545
    # neg_loss = -(-0.693147) * (1.0 - 0.8) = 0.1386
    # raw_loss = (pos_loss * pos_weight) + neg_loss
    # raw_loss = (0.5545 * 2.0) + 0.1386 = 1.109 + 0.1386 = 1.2476
    # num_pos = y.sum() = 0.8. Since it is < 1.0, clamp(min=1.0) makes it 1.0.
    # weighted_loss = (1.2476 / 1.0) * loss_weight = 1.2476 * 1.0 = 1.2476

    expected_loss_val = ( -torch.log(torch.tensor(0.5)) * 0.8 * 2.0 ) + ( -torch.log(torch.tensor(0.5)) * 0.2 )
    
    result = engine(predictions, master_tensor)
    
    assert torch.allclose(result.total_loss, expected_loss_val, atol=1e-4)

def test_engine_handles_missing_or_disabled_tasks(task_a, task_b, batch_size, seq_len):
    """
    Ensures the engine gracefully skips tasks that are either disabled in config
    or missing from the model's output dictionary.
    """
    task_b.enabled = False # Disable task B

    engine = HmoeLossEngine(tasks=[task_a, task_b])

    # Model only predicts task A
    logits_a = torch.randn(batch_size, seq_len, 2)
    predictions = MockHmoeOutput(
        task_logits={"task_a": MockHmoeTensor(logits_a)}, 
        routing_loss=MockHmoeTensor(torch.tensor(0.0))
    )
    master_tensor = MockHmoeTensor(torch.rand(batch_size, seq_len, 1))

    result = engine(predictions, master_tensor)

    # Only task A should be in the metrics log
    assert "task_a" in result.task_metrics
    assert "task_b" not in result.task_metrics

def test_engine_routing_penalty_integration(task_a, dummy_routing_loss):
    """Verifies that the routing penalty is scaled and added to the total loss."""
    routing_weight = 0.1
    engine = HmoeLossEngine(tasks=[task_a], routing_penalty_weight=routing_weight)

    logits = torch.randn(1, 1, 2)
    targets = torch.rand(1, 1, 1)

    predictions = MockHmoeOutput(
        task_logits={"task_a": MockHmoeTensor(logits)},
        routing_loss=MockHmoeTensor(dummy_routing_loss)
    )
    master_tensor = MockHmoeTensor(targets)

    result = engine(predictions, master_tensor)

    # The expected penalty component
    expected_penalty = dummy_routing_loss.item() * routing_weight

    assert result.task_metrics['routing_penalty'] == pytest.approx(expected_penalty, rel=1e-5)
    
    # Total loss should be Task A's loss + Expected Penalty
    # We can reverse engineer Task A's loss by subtracting the penalty
    derived_task_loss = result.total_loss.item() - result.task_metrics['routing_penalty']
    
    assert derived_task_loss == pytest.approx(result.task_metrics['task_a'], rel=1e-5)

def test_engine_zero_division_protection(task_a):
    """
    Validates that a batch with absolutely zero positive targets does not
    cause a NaN loss due to division by zero in the normalization step.
    """
    engine = HmoeLossEngine(tasks=[task_a])
    
    logits = torch.randn(2, 5, 2)
    # ALL targets are exactly zero
    targets = torch.zeros(2, 5, 1)

    predictions = MockHmoeOutput(
        task_logits={"task_a": MockHmoeTensor(logits)},
        routing_loss=MockHmoeTensor(torch.tensor(0.0))
    )
    master_tensor = MockHmoeTensor(targets)

    # If clamp(min=1.0) is working, this will not throw a NaN error
    result = engine(predictions, master_tensor)
    
    assert not torch.isnan(result.total_loss)