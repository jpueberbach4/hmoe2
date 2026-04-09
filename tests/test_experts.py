import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Assume the code is in hmoe2/expert.py
from hmoe2.experts import HmoeExpert

# ==========================================
# MOCKING THE HMOE2 ECOSYSTEM
# ==========================================
# Since we are testing the Expert in isolation, we mock its dependencies
# to ensure our tests are purely evaluating the Expert's internal logic.

class MockFeature:
    def __init__(self, name):
        self.name = name
    def serialize(self):
        return {"name": self.name}

class MockTask:
    def __init__(self, name):
        self.name = name

class MockHmoeInput:
    def __init__(self, tensor_data):
        self.tensor_data = tensor_data
    def get_subset(self, features):
        return self
    def to_tensor(self):
        return self.tensor_data

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def dummy_features():
    return [MockFeature("price"), MockFeature("volume")]

@pytest.fixture
def dummy_tasks():
    return [MockTask("direction_prediction"), MockTask("volatility_estimation")]

@pytest.fixture
def input_dim(dummy_features):
    return len(dummy_features)

@pytest.fixture
def hidden_dim():
    return 64

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def dummy_tensor(batch_size, seq_len, input_dim):
    return torch.randn(batch_size, seq_len, input_dim)

@pytest.fixture
def dummy_payload(dummy_tensor):
    return MockHmoeInput(dummy_tensor)


# ==========================================
# INITIALIZATION & BACKEND ROUTING TESTS
# ==========================================

def test_expert_initialization_linear(dummy_tasks, dummy_features, hidden_dim):
    """Validates that the expert correctly assumes a Linear form when asked."""
    expert = HmoeExpert(
        name="finance_expert",
        tasks=dummy_tasks,
        features=dummy_features,
        backend="LINEAR",
        hidden_dim=hidden_dim
    )
    assert expert.name == "finance_expert"
    assert expert.input_dim == len(dummy_features)
    assert expert.backend_type == "LINEAR"
    # Ensure the core was instantiated properly
    assert isinstance(expert.core, nn.Module)
    # Ensure task heads dictionary is empty before linking
    assert len(expert.task_heads) == 0

def test_expert_initialization_tcn(dummy_tasks, dummy_features, hidden_dim):
    """Validates routing to a temporal convolution backend."""
    expert = HmoeExpert(
        name="time_expert",
        tasks=dummy_tasks,
        features=dummy_features,
        backend="TCN",
        hidden_dim=hidden_dim
    )
    assert expert.backend_type == "TCN"
    assert isinstance(expert.core, nn.Module)

def test_expert_initialization_invalid_backend(dummy_tasks, dummy_features, hidden_dim):
    """Validates that the expert refuses to exist if given an impossible form."""
    with pytest.raises(ValueError, match="Unknown backend type: NONEXISTENT"):
        HmoeExpert(
            name="flawed_expert",
            tasks=dummy_tasks,
            features=dummy_features,
            backend="NONEXISTENT",
            hidden_dim=hidden_dim
        )


# ==========================================
# MULTIHEAD DYNAMIC CONFIGURATION TESTS
# ==========================================

def test_gather_tasks(dummy_tasks, dummy_features):
    """Validates that the expert accurately registers its designated tasks."""
    expert = HmoeExpert(name="test_expert", tasks=dummy_tasks, features=dummy_features)
    
    shared_task_dict = {}
    expert._gather_tasks(shared_task_dict)
    
    assert len(shared_task_dict) == 2
    assert "direction_prediction" in shared_task_dict
    assert "volatility_estimation" in shared_task_dict

@patch('hmoe2.experts.HmoeHead') # Assuming HmoeHead is imported in the same file
def test_link_tasks(mock_head_class, dummy_tasks, dummy_features, hidden_dim):
    """Validates the dynamic creation of multiple output heads for feature sharing."""
    # Setup mock to return a simple dummy module when HmoeHead is instantiated
    mock_head_class.return_value = nn.Linear(hidden_dim, 1) 

    expert = HmoeExpert(name="multihead_expert", tasks=dummy_tasks, features=dummy_features, hidden_dim=hidden_dim)
    
    # Trigger the dynamic head creation
    expert.link_tasks(global_tasks=dummy_tasks)
    
    assert len(expert.task_heads) == 2
    assert "direction_prediction" in expert.task_heads
    assert "volatility_estimation" in expert.task_heads
    
    # Verify the mock was called correctly for each task
    assert mock_head_class.call_count == 2


# ==========================================
# FORWARD PASS & LOGIT DICTIONARY TESTS
# ==========================================
@patch('hmoe2.experts.HmoeOutput')
@patch('hmoe2.experts.HmoeTensor')
def test_forward_pass_execution(mock_hmoe_tensor, mock_hmoe_output, dummy_tasks, dummy_features, dummy_payload, hidden_dim):
    """
    Validates the complete lifecycle of a thought:
    From raw feature subset -> Backend processing -> Multihead projection -> Logit dictionary.
    """
    # 1. Define a tiny custom nn.Module to act as a universal mock
    class DummyModule(nn.Module):
        def __init__(self, return_value):
            super().__init__()
            self.return_value = return_value
            self.called = False

        def forward(self, *args, **kwargs):
            self.called = True
            return self.return_value

    expert = HmoeExpert(name="forward_expert", tasks=dummy_tasks, features=dummy_features, hidden_dim=hidden_dim)

    # 2. Inject our PyTorch-native dummy modules for the multihead projections
    head_dir = DummyModule(torch.tensor([1.0]))
    head_vol = DummyModule(torch.tensor([0.5]))

    expert.task_heads["direction_prediction"] = head_dir
    expert.task_heads["volatility_estimation"] = head_vol

    # 3. Use the exact same DummyModule to mock the core backend!
    dummy_hidden_state = torch.zeros(4, 10, hidden_dim)
    expert.core = DummyModule(dummy_hidden_state)

    # Execute the forward pass
    result = expert.forward(dummy_payload)

    # Verify the backend core was engaged
    assert expert.core.called is True

    # Verify both task heads were queried
    assert head_dir.called is True
    assert head_vol.called is True

    # Extract the arguments passed to HmoeOutput instantiation
    call_args, call_kwargs = mock_hmoe_output.call_args
    assert "task_logits" in call_kwargs

    # Assert keys are present in the final dictionary of logits
    assert "direction_prediction" in call_kwargs["task_logits"]
    assert "volatility_estimation" in call_kwargs["task_logits"]


# ==========================================
# STATE SERIALIZATION TESTS
# ==========================================

def test_serialize_node(dummy_tasks, dummy_features):
    """Validates that the expert's identity and configuration can be perfectly archived."""
    expert = HmoeExpert(
        name="archive_expert", 
        tasks=dummy_tasks, 
        features=dummy_features, 
        backend="LINEAR", 
        hidden_dim=32
    )
    
    # Safely handle the dilations attribute which might not exist for LINEAR
    expert.dilations = None 

    serialized = expert._serialize_node()
    
    assert serialized["name"] == "archive_expert"
    assert serialized["type"] == "EXPERT"
    assert serialized["backend"] == "LINEAR"
    assert serialized["hidden_dim"] == 32
    assert serialized["allowed_tasks"] == ["direction_prediction", "volatility_estimation"]
    assert serialized["features"] == [{"name": "price"}, {"name": "volume"}]