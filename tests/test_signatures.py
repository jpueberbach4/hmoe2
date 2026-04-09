import pytest
import torch
from unittest.mock import patch

# Try to import signatory for test skipping decorators
try:
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False

from hmoe2.backends import SignatureBackend


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 15

@pytest.fixture
def input_dim():
    return 4

@pytest.fixture
def hidden_dim():
    return 16

@pytest.fixture
def default_config():
    return {
        'depth': 2,
        'dropout': 0.1,
        'window_length': 10
    }

@pytest.fixture
def dummy_input(batch_size, seq_len, input_dim):
    """Standard normally distributed tensor simulating sequential data."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, input_dim)


# ==========================================
# IMPORT & INITIALIZATION TESTS
# ==========================================

def test_missing_signatory_raises_error(input_dim, hidden_dim, monkeypatch):
    """
    Validates that the backend gracefully catches missing dependencies
    and raises an informative ImportError rather than crashing obscurely.
    """
    import sys
    from unittest.mock import patch

    # Dynamically grab the exact module string where SignatureBackend lives 
    # (e.g., 'hmoe2.backends' or 'hmoe2.backends.signatures')
    backend_module = SignatureBackend.__module__

    # 1. Patch the module-level reference to None (handles global imports)
    with patch(f'{backend_module}.signatory', None, create=True):
        # 2. Patch sys.modules to None (handles lazy/local imports inside __init__)
        with patch.dict('sys.modules', {'signatory': None}):
            
            with pytest.raises(ImportError, match="The 'signatory' library is required"):
                SignatureBackend(input_dim, hidden_dim)


@pytest.mark.skipif(not HAS_SIGNATORY, reason="Requires 'signatory' library")
def test_signature_initialization_and_config(input_dim, hidden_dim, default_config):
    """Validates configuration parsing and proper Signature dimension calculation."""
    model = SignatureBackend(input_dim, hidden_dim, config=default_config)
    
    assert model.depth == default_config['depth']
    assert model.window_length == default_config['window_length']
    
    # Calculate expected signature channels
    # For depth d and input channels c, the number of signature terms is:
    # c + c^2 + ... + c^d
    # With input_dim=4, depth=2: 4 + 4^2 = 20
    expected_sig_channels = signatory.signature_channels(input_dim, default_config['depth'])
    
    assert model.sig_channels == expected_sig_channels
    assert model.net[0].in_features == expected_sig_channels


# ==========================================
# FORWARD PASS & SHAPE TESTS
# ==========================================

@pytest.mark.skipif(not HAS_SIGNATORY, reason="Requires 'signatory' library")
def test_signature_forward_shape(dummy_input, input_dim, hidden_dim, default_config):
    """Validates the complete forward pass outputs the correctly dimensioned hidden state."""
    model = SignatureBackend(input_dim, hidden_dim, config=default_config)
    
    out = model(dummy_input)
    
    # Output must match [Batch, Sequence, Hidden]
    assert out.shape == (dummy_input.size(0), dummy_input.size(1), hidden_dim)


@pytest.mark.skipif(not HAS_SIGNATORY, reason="Requires 'signatory' library")
def test_short_sequence_handling(input_dim, hidden_dim, default_config):
    """
    Validates that the replicate padding correctly handles sequences that 
    are shorter than the specified lookback window_length.
    """
    # Window length is 10, but we feed a sequence of length 3
    short_seq_len = 3
    model = SignatureBackend(input_dim, hidden_dim, config=default_config)
    
    short_input = torch.randn(2, short_seq_len, input_dim)
    
    # Should not crash during the unfold operation
    out = model(short_input)
    
    # Ensure standard output bounds
    assert out.shape == (2, short_seq_len, hidden_dim)


# ==========================================
# CAUSALITY TESTS
# ==========================================

@pytest.mark.skipif(not HAS_SIGNATORY, reason="Requires 'signatory' library")
def test_strict_causality_rolling_window(input_dim, hidden_dim, default_config):
    """
    Validates mathematical causality. Modifying a future timestep MUST NOT 
    alter the calculated signature (and thus hidden state) of past timesteps.
    """
    seq_len = 20
    model = SignatureBackend(input_dim, hidden_dim, config=default_config)
    model.eval()

    x1 = torch.randn(1, seq_len, input_dim)
    x2 = x1.clone()
    
    # Modify the sequence at timestep 10
    x2[:, 10:, :] = torch.randn(1, seq_len - 10, input_dim)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Signatures for timesteps 0 through 9 should remain completely identical
    assert torch.allclose(out1[:, :10, :], out2[:, :10, :], atol=1e-5), \
        "Causality leak detected: Future data influenced past signature calculations."
    
    # Signatures for timestep 10 and beyond should diverge due to the new data
    assert not torch.allclose(out1[:, 10:, :], out2[:, 10:, :], atol=1e-5)