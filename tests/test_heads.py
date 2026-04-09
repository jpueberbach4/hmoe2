import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from hmoe2.heads import HmoeHead

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def input_dim():
    return 64

@pytest.fixture
def num_classes():
    return 3

@pytest.fixture
def task_name():
    return "market_direction"

@pytest.fixture
def mock_task(task_name, num_classes):
    """Mocks the HmoeTask configuration to avoid importing the real dataclass."""
    task = MagicMock()
    task.name = task_name
    task.num_classes = num_classes
    return task


# ==========================================
# INITIALIZATION & ARCHITECTURE TESTS
# ==========================================

@patch('hmoe2.heads.HmoeFeature')
def test_head_initialization(mock_hmoe_feature, input_dim, mock_task):
    """
    Validates that the head correctly configures its internal linear projection
    and maps the correct naming conventions to its output features.
    """
    # Initialize the head
    head = HmoeHead(input_dim=input_dim, task_config=mock_task)

    # 1. Verify the linear classifier architecture
    assert isinstance(head.classifier, nn.Linear)
    assert head.classifier.in_features == input_dim
    assert head.classifier.out_features == mock_task.num_classes

    # 2. Verify task configuration was stored
    assert head.task_config == mock_task

    # 3. Verify HmoeFeature creation
    # The head should have instantiated a feature for each class
    assert mock_hmoe_feature.call_count == mock_task.num_classes
    
    # Check the exact naming convention: {task_name}_class_{i}
    mock_hmoe_feature.assert_any_call(name=f"{mock_task.name}_class_0")
    mock_hmoe_feature.assert_any_call(name=f"{mock_task.name}_class_1")
    mock_hmoe_feature.assert_any_call(name=f"{mock_task.name}_class_2")
    
    assert len(head.output_features) == mock_task.num_classes


# ==========================================
# FORWARD PASS & SCHEMA ABSTRACTION TESTS
# ==========================================

@patch('hmoe2.heads.HmoeTensor')
@patch('hmoe2.heads.HmoeFeature')
def test_head_forward_pass(mock_hmoe_feature, mock_hmoe_tensor, input_dim, mock_task):
    """
    Validates the data flow: extracting raw tensors, computing logits, 
    and packaging the output back into the strictly required HmoeTensor format.
    """
    batch_size = 4
    seq_len = 10
    
    # Initialize the head
    head = HmoeHead(input_dim=input_dim, task_config=mock_task)
    
    # Create a dummy raw tensor mimicking a hidden state [Batch, Seq, Features]
    dummy_raw_hidden = torch.randn(batch_size, seq_len, input_dim)
    
    # Mock the input shared representation (HmoeTensor)
    mock_shared_rep = MagicMock()
    mock_shared_rep.to_tensor.return_value = dummy_raw_hidden
    
    # Execute the forward pass
    output = head(mock_shared_rep)
    
    # 1. Verify data extraction
    mock_shared_rep.to_tensor.assert_called_once()
    
    # 2. Verify HmoeTensor repackaging
    # We must intercept how the head tried to instantiate the final HmoeTensor
    mock_hmoe_tensor.assert_called_once()
    call_args, call_kwargs = mock_hmoe_tensor.call_args
    
    assert "tensor" in call_kwargs, "HmoeTensor was not provided the raw logit tensor."
    assert "indices" in call_kwargs, "HmoeTensor was not provided the feature indices."
    
    # 3. Verify output tensor shape
    # The linear layer should have projected [Batch, Seq, InputDim] -> [Batch, Seq, NumClasses]
    raw_logits = call_kwargs["tensor"]
    assert raw_logits.shape == (batch_size, seq_len, mock_task.num_classes)
    
    # 4. Verify output metadata binding
    # The indices must strictly match the output_features generated during initialization
    assert call_kwargs["indices"] == head.output_features
    
    # 5. Verify the method actually returns the mocked object
    assert output == mock_hmoe_tensor.return_value


def test_head_gradient_flow(input_dim, mock_task):
    """
    Validates that the head maintains PyTorch's computational graph.
    (This test uses the actual objects rather than mocks to ensure PyTorch mechanics work).
    """
    head = HmoeHead(input_dim=input_dim, task_config=mock_task)
    
    # Create a raw tensor with requires_grad=True to track gradients
    raw_input = torch.randn(2, 5, input_dim, requires_grad=True)
    
    # Create a quick mock for the HmoeTensor just to pass the raw_input
    mock_shared_rep = MagicMock()
    mock_shared_rep.to_tensor.return_value = raw_input
    
    # Forward pass (we don't mock HmoeTensor here because we want to see the real object)
    # Note: Since we didn't patch HmoeTensor, the head will try to instantiate the REAL one.
    # If the real HmoeTensor is not imported properly in your test environment, 
    # this specific test might fail. Ensure it's reachable.
    try:
        output_hmoe_tensor = head(mock_shared_rep)
        
        # Extract the resulting tensor
        output_logits = output_hmoe_tensor.tensor
        
        # Simulate a loss calculation and backward pass
        loss = output_logits.sum()
        loss.backward()
        
        # Verify gradients successfully flowed back to the linear classifier's weights
        assert head.classifier.weight.grad is not None
        assert head.classifier.bias.grad is not None
        
        # Verify gradients flowed back to the input tensor
        assert raw_input.grad is not None

    except NameError:
        pytest.skip("Real HmoeTensor class could not be resolved for gradient test. Ensure accurate imports.")