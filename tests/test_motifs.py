import pytest
import torch

from hmoe2.backends import MotifsBackend

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 20

@pytest.fixture
def input_dim():
    return 8

@pytest.fixture
def hidden_dim():
    return 16

@pytest.fixture
def default_config():
    return {
        'num_motifs': 5,
        'motif_length': 10,
        'dropout': 0.1
    }

@pytest.fixture
def dummy_input(batch_size, seq_len, input_dim):
    """Standard normal input tensor."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, input_dim)


# ==========================================
# INITIALIZATION & CONFIGURATION TESTS
# ==========================================

def test_motifs_initialization_and_shape(dummy_input, input_dim, hidden_dim, default_config):
    """Validates module initialization and output tensor dimensions."""
    model = MotifsBackend(input_dim, hidden_dim, config=default_config)
    
    # Check parameter registration
    assert model.motifs.shape == (default_config['num_motifs'], input_dim, default_config['motif_length'])
    
    out = model(dummy_input)
    
    # Output should project sequence to hidden_dim
    assert out.shape == (dummy_input.size(0), dummy_input.size(1), hidden_dim)

def test_config_parsing_bug_catcher(input_dim, hidden_dim):
    """
    Validates that config values are parsed to the correct attributes.
    WARNING: This test will fail on your current implementation because 
    motif_length is mistakenly extracting 'num_motifs' from the config dictionary.
    """
    config = {'num_motifs': 3, 'motif_length': 7, 'dropout': 0.5}
    model = MotifsBackend(input_dim, hidden_dim, config=config)
    
    assert model.num_motifs == 3
    assert model.motif_length == 7, "Bug detected: motif_length is not parsing correctly from config."


# ==========================================
# MATHEMATICAL & STABILITY TESTS
# ==========================================

def test_zero_variance_nan_protection(input_dim, hidden_dim, default_config, batch_size, seq_len):
    """
    Ensures that input tensors with zero variance (all constants) do not
    result in NaN values during Z-normalization due to division by zero.
    """
    model = MotifsBackend(input_dim, hidden_dim, config=default_config)
    model.eval()
    
    # Input of all ones (variance = 0)
    flat_input = torch.ones(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        out = model(flat_input)
        
    assert not torch.isnan(out).any(), "NaNs detected! The 1e-8 epsilon failed to prevent division by zero."

def test_forward_call_counter(dummy_input, input_dim, hidden_dim, default_config):
    """Verifies that the internal debugging metric accurately tracks inferences."""
    model = MotifsBackend(input_dim, hidden_dim, config=default_config)
    
    assert model._forward_calls == 0
    model(dummy_input)
    assert model._forward_calls == 1
    model(dummy_input)
    assert model._forward_calls == 2


# ==========================================
# CAUSALITY & SEQUENCE TESTS
# ==========================================

def test_strict_causality(input_dim, hidden_dim, default_config):
    """
    Validates that the sliding window padding is strictly causal.
    Modifying a future timestep MUST NOT alter the output of past timesteps.
    """
    seq_len = 15
    model = MotifsBackend(input_dim, hidden_dim, config=default_config)
    model.eval()

    x1 = torch.randn(1, seq_len, input_dim)
    x2 = x1.clone()
    
    # Modify the sequence at timestep 8
    x2[:, 8:, :] = torch.randn(1, seq_len - 8, input_dim)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Outputs should be identical up to the modification point
    assert torch.allclose(out1[:, :8, :], out2[:, :8, :], atol=1e-5), "Causality leak detected: Future data influenced past outputs."
    
    # Outputs should diverge after the modification point
    assert not torch.allclose(out1[:, 8:, :], out2[:, 8:, :], atol=1e-5)

def test_short_sequence_handling(input_dim, hidden_dim, default_config):
    """
    Validates that the padding strategy correctly handles sequences that are 
    shorter than the motif length itself.
    """
    # motif_length is 10, sequence length is 4
    short_seq_len = 4
    model = MotifsBackend(input_dim, hidden_dim, config=default_config)
    
    short_input = torch.randn(2, short_seq_len, input_dim)
    
    # Should execute without dimension mismatch errors from unfold
    out = model(short_input)
    
    assert out.shape == (2, short_seq_len, hidden_dim)