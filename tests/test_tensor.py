import pytest
import torch
import math
from dataclasses import FrozenInstanceError

# NOTE: Adjust this import to match the actual location of your DTOs
from hmoe2.schema import HmoeFeature
from hmoe2.tensor import HmoeTensor, HmoeInput, HmoeOutput 


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def feature_a():
    return HmoeFeature(name="price")

@pytest.fixture
def feature_b():
    return HmoeFeature(name="volume")

@pytest.fixture
def feature_c():
    return HmoeFeature(name="momentum")

@pytest.fixture
def sample_features(feature_a, feature_b, feature_c):
    return [feature_a, feature_b, feature_c]

@pytest.fixture
def base_dict_data():
    """Standard dictionary with clean float/int data."""
    return {
        "price": [10.5, 11.0, 10.8],
        "volume": [100, 200, 150],
        "momentum": [0.1, 0.5, -0.2]
    }

@pytest.fixture
def dirty_dict_data():
    """Dictionary containing invalid types, NaNs, and infinities."""
    return {
        "price": [10.5, "invalid", 10.8],
        "volume": [100, None, float('inf')],
        "momentum": [0.1, float('nan'), -0.2]
    }

@pytest.fixture
def sample_hmoe_tensor(base_dict_data):
    return HmoeTensor.from_dict(base_dict_data)


# ==========================================
# HMOE TENSOR TESTS
# ==========================================

def test_from_dict_clean_data(base_dict_data):
    """Validates that clean dictionary data is accurately parsed into the correct shape."""
    h_tensor = HmoeTensor.from_dict(base_dict_data)
    
    # Check basic types
    assert isinstance(h_tensor, HmoeTensor)
    assert isinstance(h_tensor.tensor, torch.Tensor)
    
    # Check Feature Metadata
    assert len(h_tensor.indices) == 3
    assert h_tensor.indices[0].name == "price"
    assert h_tensor.indices[1].name == "volume"
    
    # Check Shape: 3 keys (features) x 3 values (sequence) -> [1, 3, 3] (batch, seq, feat)
    assert h_tensor.tensor.shape == (1, 3, 3)
    
    # Check actual values (sequence 0, feature 0 -> 10.5)
    assert torch.isclose(h_tensor.tensor[0, 0, 0], torch.tensor(10.5))

def test_from_dict_dirty_data_handling(dirty_dict_data):
    """Validates that strings, None, NaN, and Inf are handled and zeroed out safely."""
    h_tensor = HmoeTensor.from_dict(dirty_dict_data)
    
    # The second sequence element has 'invalid', None, and NaN. 
    # All should be converted to 0.0 by nan_to_num.
    assert h_tensor.tensor[0, 1, 0] == 0.0  # "invalid" -> 0.0
    assert h_tensor.tensor[0, 1, 1] == 0.0  # None -> 0.0
    assert h_tensor.tensor[0, 1, 2] == 0.0  # NaN -> 0.0
    
    # The third sequence element for volume has 'inf'. Should be 0.0.
    assert h_tensor.tensor[0, 2, 1] == 0.0  # inf -> 0.0

def test_hmoetensor_immutability(sample_hmoe_tensor):
    """Validates that the frozen dataclass prevents arbitrary attribute mutation."""
    with pytest.raises(FrozenInstanceError):
        sample_hmoe_tensor.indices = []

def test_get_indices(sample_hmoe_tensor, sample_features):
    """Validates get_indices returns the proper feature list."""
    indices = sample_hmoe_tensor.get_indices()
    assert len(indices) == 3
    assert [f.name for f in indices] == [f.name for f in sample_features]

def test_get_subset_success(sample_hmoe_tensor, feature_b, feature_a):
    """Validates slicing the tensor by feature names."""
    # Request out of order to ensure exact name mapping works
    subset = sample_hmoe_tensor.get_subset([feature_b, feature_a])
    
    assert len(subset.indices) == 2
    assert subset.indices[0].name == "volume"
    assert subset.indices[1].name == "price"
    
    # Original shape was (1, 3, 3). Subset should be (1, 3, 2)
    assert subset.tensor.shape == (1, 3, 2)
    
    # Validate data matches the reordered slice
    assert torch.allclose(subset.tensor[0, :, 0], sample_hmoe_tensor.tensor[0, :, 1]) # volume
    assert torch.allclose(subset.tensor[0, :, 1], sample_hmoe_tensor.tensor[0, :, 0]) # price

def test_get_subset_missing_feature(sample_hmoe_tensor, feature_a):
    """Validates an explicit ValueError is raised if a feature is missing."""
    missing_feature = HmoeFeature(name="non_existent")
    
    with pytest.raises(ValueError, match="Feature translation failed"):
        sample_hmoe_tensor.get_subset([feature_a, missing_feature])

def test_to_tensor_extraction(sample_hmoe_tensor):
    """Validates extraction of the raw underlying tensor."""
    raw = sample_hmoe_tensor.to_tensor()
    assert isinstance(raw, torch.Tensor)
    assert raw.shape == sample_hmoe_tensor.tensor.shape


# ==========================================
# HMOE INPUT TESTS
# ==========================================

@pytest.fixture
def sample_hmoe_input(sample_hmoe_tensor):
    return HmoeInput(tensor=sample_hmoe_tensor)

def test_hmoeinput_subset_delegation(sample_hmoe_input, feature_a):
    """Validates HmoeInput correctly delegates the subset operation to HmoeTensor."""
    subset_input = sample_hmoe_input.get_subset([feature_a])
    
    assert isinstance(subset_input, HmoeInput)
    assert isinstance(subset_input.tensor, HmoeTensor)
    assert len(subset_input.tensor.indices) == 1
    assert subset_input.tensor.indices[0].name == "price"

def test_hmoeinput_to_tensor(sample_hmoe_input):
    """Validates HmoeInput correctly exposes the underlying raw tensor."""
    raw = sample_hmoe_input.to_tensor()
    assert isinstance(raw, torch.Tensor)


# ==========================================
# HMOE OUTPUT TESTS
# ==========================================

@pytest.fixture
def sample_hmoe_output(sample_hmoe_tensor):
    return HmoeOutput(
        task_logits={"task_1": sample_hmoe_tensor, "task_2": sample_hmoe_tensor}
    )

def test_hmoeoutput_default_routing_loss(sample_hmoe_output):
    """Validates the default routing loss is instantiated as a 0.0 scalar HmoeTensor."""
    loss = sample_hmoe_output.routing_loss
    
    assert isinstance(loss, HmoeTensor)
    assert loss.indices == []
    assert loss.tensor.item() == 0.0

def test_hmoeoutput_custom_routing_loss():
    """Validates custom routing loss is retained."""
    custom_loss_tensor = HmoeTensor(tensor=torch.tensor([1.5]), indices=[])
    output = HmoeOutput(
        task_logits={}, 
        routing_loss=custom_loss_tensor
    )
    
    assert output.routing_loss.tensor.item() == 1.5


# ==========================================
# DEVICE PLACEMENT TESTS (.to)
# ==========================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfers_cuda(sample_hmoe_tensor, sample_hmoe_input, sample_hmoe_output):
    """Validates .to(device) accurately moves data and returns new instances without mutating originals."""
    device = torch.device('cuda:0')
    
    # HmoeTensor
    gpu_tensor = sample_hmoe_tensor.to(device)
    assert gpu_tensor.tensor.device.type == 'cuda'
    assert sample_hmoe_tensor.tensor.device.type == 'cpu' # Original unchanged
    
    # HmoeInput
    gpu_input = sample_hmoe_input.to(device)
    assert gpu_input.tensor.tensor.device.type == 'cuda'
    
    # HmoeOutput
    gpu_output = sample_hmoe_output.to(device)
    assert gpu_output.task_logits["task_1"].tensor.device.type == 'cuda'
    assert gpu_output.task_logits["task_2"].tensor.device.type == 'cuda'
    assert gpu_output.routing_loss.tensor.device.type == 'cuda'

def test_device_transfers_cpu(sample_hmoe_tensor, sample_hmoe_output):
    """Validates .to() works locally even if CUDA is absent."""
    device = torch.device('cpu')
    
    cpu_tensor = sample_hmoe_tensor.to(device)
    assert cpu_tensor.tensor.device.type == 'cpu'
    
    cpu_output = sample_hmoe_output.to(device)
    assert cpu_output.routing_loss.tensor.device.type == 'cpu'