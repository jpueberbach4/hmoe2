import pytest
import torch
from unittest.mock import MagicMock

from hmoe2.gates import HmoeGate, HmoeGateTCN, HmoeGateTopK, HmoeGateGRU
from hmoe2.tensor import HmoeTensor

# ==========================================
# FIXTURES & MOCKS
# ==========================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 12

@pytest.fixture
def input_dim():
    return 16

@pytest.fixture
def num_children():
    return 5

@pytest.fixture
def dummy_raw_tensor(batch_size, seq_len, input_dim):
    """Generates standard normally distributed input data."""
    torch.manual_seed(42) # For reproducibility in noise tests
    return torch.randn(batch_size, seq_len, input_dim)

@pytest.fixture
def dummy_payload(dummy_raw_tensor):
    """Mocks the HmoeTensor wrapper to isolate the gate logic."""
    mock_payload = MagicMock()
    mock_payload.to_tensor.return_value = dummy_raw_tensor
    return mock_payload


# ==========================================
# HMOEGATE (LINEAR) TESTS
# ==========================================

def test_hmoegate_shape_and_distribution(dummy_payload, input_dim, num_children, batch_size, seq_len):
    """Validates the standard linear gate outputs proper probabilities."""
    gate = HmoeGate(input_dim, num_children)
    weights = gate(dummy_payload)

    # Output shape should be [Batch, Sequence, NumChildren]
    assert weights.shape == (batch_size, seq_len, num_children)

    # Probabilities across children must sum to 1.0 for every timestep
    sums = weights.sum(dim=-1)
    expected_sums = torch.ones(batch_size, seq_len)
    assert torch.allclose(sums, expected_sums, atol=1e-5), "Routing probabilities do not sum to 1."


# ==========================================
# HMOEGATE TCN TESTS
# ==========================================

def test_hmoegate_tcn_shape_and_distribution(dummy_payload, input_dim, num_children, batch_size, seq_len):
    """Validates the TCN gate output constraints."""
    # Using a small dilation config to keep the test fast
    gate = HmoeGateTCN(input_dim, num_children, hidden_dim=8, dilations=[1, 2])
    weights = gate(dummy_payload)

    assert weights.shape == (batch_size, seq_len, num_children)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

def test_hmoegate_tcn_noise_behavior(dummy_payload, input_dim, num_children):
    """Ensures noise is only injected during training, not evaluation."""
    gate = HmoeGateTCN(input_dim, num_children, hidden_dim=8, dilations=[1, 2], noise_std=0.5)
    
    # Run in eval mode (deterministic)
    gate.eval()
    with torch.no_grad():
        eval_weights_1 = gate(dummy_payload)
        eval_weights_2 = gate(dummy_payload)
    
    # Outputs should be identical in eval mode
    assert torch.allclose(eval_weights_1, eval_weights_2, atol=1e-6)

    # Run in train mode (stochastic)
    gate.train()
    with torch.no_grad():
        train_weights_1 = gate(dummy_payload)
        train_weights_2 = gate(dummy_payload)
    
    # Outputs should diverge due to noise injection
    assert not torch.allclose(train_weights_1, train_weights_2, atol=1e-6)


# ==========================================
# HMOEGATE TOP-K (SPARSE) TESTS
# ==========================================

@pytest.mark.parametrize("k", [1, 2, 4])
def test_hmoegate_topk_sparsity(k, dummy_payload, input_dim, num_children, batch_size, seq_len):
    """Validates that exactly K experts are selected and probabilities sum to 1."""
    gate = HmoeGateTopK(input_dim, num_children, k=k, noise_std=0.0)
    weights = gate(dummy_payload)

    assert weights.shape == (batch_size, seq_len, num_children)
    
    # Check that probabilities sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    # Check sparsity: Exactly 'k' elements should be strictly greater than 0
    # (using > 1e-6 to account for floating point inaccuracies)
    non_zero_counts = (weights > 1e-6).sum(dim=-1)
    expected_counts = torch.full((batch_size, seq_len), k)
    
    assert torch.equal(non_zero_counts, expected_counts), f"Expected exactly {k} non-zero routing weights."

def test_hmoegate_topk_clamping(dummy_payload, input_dim, num_children):
    """Validates that K is safely clamped if it exceeds the number of children."""
    excessive_k = num_children + 5
    gate = HmoeGateTopK(input_dim, num_children, k=excessive_k)
    
    # The gate should have clamped internal k to num_children
    assert gate.k == num_children


# ==========================================
# HMOEGATE GRU TESTS
# ==========================================

def test_hmoegate_gru_shape_and_distribution(dummy_payload, input_dim, num_children, batch_size, seq_len):
    """Validates GRU gate output constraints."""
    gate = HmoeGateGRU(input_dim, num_children, hidden_dim=16, num_layers=1)
    weights = gate(dummy_payload)

    assert weights.shape == (batch_size, seq_len, num_children)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

def test_hmoegate_gru_dropout_config(input_dim, num_children):
    """Validates that dropout is correctly handled based on layer count."""
    # Single layer should force GRU dropout to 0 internally to prevent PyTorch warnings
    gate_single = HmoeGateGRU(input_dim, num_children, num_layers=1, dropout_p=0.5)
    assert gate_single.gru.dropout == 0.0

    # Multi-layer should apply the requested dropout
    gate_multi = HmoeGateGRU(input_dim, num_children, num_layers=2, dropout_p=0.5)
    assert gate_multi.gru.dropout == 0.5

def test_hmoegate_gru_noise_behavior(dummy_payload, input_dim, num_children):
    """Ensures GRU noise is only injected during training."""
    gate = HmoeGateGRU(input_dim, num_children, hidden_dim=16, num_layers=1, noise_std=0.2)
    
    gate.eval()
    with torch.no_grad():
        eval_w1 = gate(dummy_payload)
        eval_w2 = gate(dummy_payload)
    assert torch.allclose(eval_w1, eval_w2, atol=1e-6)

    gate.train()
    with torch.no_grad():
        train_w1 = gate(dummy_payload)
        train_w2 = gate(dummy_payload)
    assert not torch.allclose(train_w1, train_w2, atol=1e-6)