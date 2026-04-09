import os
import json
import pytest
import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from hmoe2.nodes import HmoeNode
from hmoe2.loss import HmoeLossEngine, HmoeLossResult
from hmoe2.tensor import HmoeTensor, HmoeInput
from hmoe2.trainer import HmoeTrainer


# ==========================================
# DUMMY PYTORCH COMPONENTS FOR TESTING
# ==========================================

@dataclass
class DummyHmoeLossResult:
    """Mimics the expected HmoeLossResult schema."""
    total_loss: torch.Tensor
    task_metrics: Dict[str, float]

class DummyModel(nn.Module):
    """A lightweight linear model that mimics the HmoeNode interface."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        
    def forward(self, hmoe_input: HmoeInput):
        # Extract raw tensor from the DTO chain
        raw_tensor = hmoe_input.tensor.tensor 
        return self.layer(raw_tensor)

class DummyLossEngine(nn.Module):
    """Calculates a real MSE loss so gradients can propagate."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, clean_master: HmoeTensor):
        # Using the first 2 features of the target to match prediction shape [Batch, 2]
        target = clean_master.tensor[:, :2] 
        loss = self.mse(predictions, target)
        return DummyHmoeLossResult(
            total_loss=loss, 
            task_metrics={"dummy_task_acc": 0.95}
        )

class DummyHmoeTensor:
    """A lightweight mock of your HmoeTensor DTO."""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
    def to(self, device):
        return DummyHmoeTensor(self.tensor.to(device))


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def dummy_loss_engine():
    return DummyLossEngine()

@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.1)

@pytest.fixture
def checkpoint_dir(tmp_path):
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    return str(tmp_path / "checkpoints")

@pytest.fixture
def trainer(dummy_model, dummy_loss_engine, dummy_optimizer, device, checkpoint_dir):
    return HmoeTrainer(
        model=dummy_model,
        loss_engine=dummy_loss_engine,
        optimizer=dummy_optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler_patience=2
    )

@pytest.fixture
def dummy_dataloader():
    """Generates 3 batches of fake sequential data."""
    data = []
    for _ in range(3):
        # Shape: [Batch=2, Features=4]
        raw_tensor = torch.randn(2, 4)
        data.append(DummyHmoeTensor(raw_tensor))
    return data


# ==========================================
# INITIALIZATION TESTS
# ==========================================

def test_trainer_initialization(trainer, checkpoint_dir, device):
    """Validates that properties are set and checkpoint directory is created."""
    assert os.path.exists(checkpoint_dir)
    assert trainer.device == device
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert trainer.history["train_loss"] == []
    
    # Ensure model and loss engine were moved to the correct device
    assert next(trainer.model.parameters()).device == device


# ==========================================
# PROCESS BATCH TESTS
# ==========================================

def test_process_batch_training(trainer, dummy_dataloader):
    """Validates the training step computes gradients and updates weights."""
    batch = dummy_dataloader[0]
    
    # Store initial weights to verify they change
    initial_weight = trainer.model.layer.weight.clone()
    
    with patch.object(trainer.optimizer, 'zero_grad', wraps=trainer.optimizer.zero_grad) as mock_zero:
        with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as mock_step:
            
            result = trainer._process_batch(batch, is_training=True)
            
            # Check PyTorch mechanics were triggered
            mock_zero.assert_called_once()
            mock_step.assert_called_once()
            
            # Gradients should exist and weights should have updated
            assert trainer.model.layer.weight.grad is not None
            assert not torch.allclose(initial_weight, trainer.model.layer.weight)
            assert result.total_loss.item() > 0

def test_process_batch_evaluation(trainer, dummy_dataloader):
    """Validates the evaluation step bypasses optimization entirely."""
    batch = dummy_dataloader[0]
    initial_weight = trainer.model.layer.weight.clone()
    
    with patch.object(trainer.optimizer, 'zero_grad') as mock_zero:
        with patch.object(trainer.optimizer, 'step') as mock_step:
            
            # Note: We still need torch.no_grad() context here because the trainer 
            # relies on the fit() loop to wrap validation batches in no_grad().
            with torch.no_grad():
                trainer._process_batch(batch, is_training=False)
            
            # Optimizer should not be touched
            mock_zero.assert_not_called()
            mock_step.assert_not_called()
            
            # Weights must remain identical
            assert torch.allclose(initial_weight, trainer.model.layer.weight)


# ==========================================
# FIT LOOP & SCHEDULING TESTS
# ==========================================

def test_fit_standard_execution(trainer, dummy_dataloader):
    """Validates a standard multi-epoch run populates history and saves metrics."""
    epochs = 2
    trainer.fit(dummy_dataloader, dummy_dataloader, epochs=epochs, patience=5)
    
    assert len(trainer.history["train_loss"]) == epochs
    assert len(trainer.history["val_loss"]) == epochs
    assert len(trainer.history["task_metrics"]) == epochs
    
    # Ensure metrics JSON was written
    metrics_path = os.path.join(trainer.checkpoint_dir, "metrics_history.json")
    assert os.path.exists(metrics_path)
    
    with open(metrics_path, 'r') as f:
        saved_history = json.load(f)
        assert len(saved_history["val_loss"]) == epochs

def test_early_stopping_triggered(trainer, dummy_dataloader):
    """Validates training aborts when patience is exceeded."""
    epochs = 10
    patience = 2
    
    # Mock the _process_batch to always return an increasing validation loss
    # to artificially trigger early stopping.
    dummy_increasing_loss = DummyHmoeLossResult(
        total_loss=torch.tensor(10.0), 
        task_metrics={}
    )
    
    # Epoch 1: Loss = 10.0 (Best)
    # Epoch 2: Loss = 11.0 (Patience = 1)
    # Epoch 3: Loss = 12.0 (Patience = 2 -> STOP)
    loss_sequence = [torch.tensor(10.0), torch.tensor(11.0), torch.tensor(12.0)]
    
    def mock_process(batch, is_training):
        if not is_training:
            # Shift the next loss off the sequence
            dummy_increasing_loss.total_loss = loss_sequence.pop(0) if loss_sequence else torch.tensor(20.0)
        return dummy_increasing_loss

    trainer._process_batch = mock_process
    
    trainer.fit(dummy_dataloader, dummy_dataloader, epochs=epochs, patience=patience)
    
    # Training should have stopped at epoch 3
    assert len(trainer.history["val_loss"]) == 3

def test_lr_scheduler_plateau(trainer, dummy_dataloader):
    """Validates the learning rate slashes when the validation loss plateaus."""
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Set up a stagnant validation loss
    stagnant_loss = DummyHmoeLossResult(total_loss=torch.tensor(5.0), task_metrics={})
    trainer._process_batch = lambda batch, is_training: stagnant_loss
    
    # Run fit for slightly longer than the scheduler's patience (which is 2)
    trainer.fit(dummy_dataloader, dummy_dataloader, epochs=4, patience=10)
    
    final_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Scheduler factor is 0.5, so it should have halved the LR
    assert final_lr == initial_lr * 0.5


# ==========================================
# CHECKPOINTING TESTS
# ==========================================

def test_save_and_load_checkpoint(trainer, dummy_dataloader):
    """Validates the model state is properly serialized and restorable."""
    # Run 1 epoch to change weights and optimizer state
    trainer.fit(dummy_dataloader, dummy_dataloader, epochs=1, patience=5)
    
    # Save the current state
    target_checkpoint = "test_ckpt.pt"
    trainer.save_checkpoint(target_checkpoint)
    
    # Record the trained weights
    trained_weight = trainer.model.layer.weight.clone()
    
    # Perturb the model weights to simulate a reset or different state
    with torch.no_grad():
        trainer.model.layer.weight.fill_(0.0)
    assert not torch.allclose(trained_weight, trainer.model.layer.weight)
    
    # Reload the checkpoint
    trainer.load_checkpoint(target_checkpoint)
    
    # Verify weights are fully restored
    assert torch.allclose(trained_weight, trainer.model.layer.weight)

def test_load_checkpoint_not_found(trainer):
    """Validates a clean FileNotFoundError is raised for bad paths."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        trainer.load_checkpoint("does_not_exist.pt")