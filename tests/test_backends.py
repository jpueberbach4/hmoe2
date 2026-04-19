import pytest
import torch

from hmoe2.backends import (
     StrictCausalConv1d, LinearBackend, TcnBackend, GruBackend,
     CausalTransformerBackend, GatedResidualBackend, LstmBackend, RnnBackend
 )

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 16

@pytest.fixture
def input_dim():
    return 32

@pytest.fixture
def hidden_dim():
    return 64

@pytest.fixture
def dummy_sequence_data(batch_size, seq_len, input_dim):
    """Data shaped [Batch, Sequence, Features] for most backends."""
    return torch.randn(batch_size, seq_len, input_dim)

@pytest.fixture
def dummy_conv_data(batch_size, input_dim, seq_len):
    """Data shaped [Batch, Channels, Sequence] for StrictCausalConv1d."""
    return torch.randn(batch_size, input_dim, seq_len)


# ==========================================
# STRICT CAUSAL CONV1D TESTS
# ==========================================

def test_strict_causal_conv1d_shape(dummy_conv_data, input_dim, hidden_dim):
    """Test if StrictCausalConv1d outputs the correct shape."""
    model = StrictCausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=2)
    out = model(dummy_conv_data)
    
    # Expected shape: [Batch, OutChannels, Sequence]
    assert out.shape == (dummy_conv_data.size(0), hidden_dim, dummy_conv_data.size(2))

def test_strict_causal_conv1d_causality(input_dim, hidden_dim):
    """Test that modifying a future timestep does not affect past outputs."""
    seq_len = 10
    model = StrictCausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=2)
    model.eval() # Disable dropout/batchnorm randomness if any was added

    x1 = torch.randn(1, input_dim, seq_len)
    x2 = x1.clone()
    
    # Modify the tensor at timestep 5
    x2[:, :, 5:] = torch.randn(1, input_dim, seq_len - 5)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Outputs should be identical up to the modified timestep (timestep 4)
    assert torch.allclose(out1[:, :, :5], out2[:, :, :5], atol=1e-6)
    # Outputs should differ after the modified timestep
    assert not torch.allclose(out1[:, :, 5:], out2[:, :, 5:], atol=1e-6)


# ==========================================
# LINEAR BACKEND TESTS
# ==========================================

def test_linear_backend_shape(dummy_sequence_data, input_dim, hidden_dim):
    model = LinearBackend(input_dim, hidden_dim)
    out = model(dummy_sequence_data)
    assert out.shape == (dummy_sequence_data.size(0), dummy_sequence_data.size(1), hidden_dim)

def test_linear_backend_config(input_dim, hidden_dim):
    config = {'dropout': 0.5}
    model = LinearBackend(input_dim, hidden_dim, config)
    # Verify dropout was set properly in the sequential block
    dropouts = [m for m in model.net.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 2
    assert dropouts[0].p == 0.5


# ==========================================
# TCN BACKEND TESTS
# ==========================================

def test_tcn_backend_shape(dummy_sequence_data, input_dim, hidden_dim):
    model = TcnBackend(input_dim, hidden_dim)
    out = model(dummy_sequence_data)
    assert out.shape == (dummy_sequence_data.size(0), dummy_sequence_data.size(1), hidden_dim)

def test_tcn_backend_causality(input_dim, hidden_dim):
    """Test strict causality across the entire TCN stack."""
    seq_len = 20
    model = TcnBackend(input_dim, hidden_dim, config={'dilations': [1, 2, 4]})
    model.eval()

    x1 = torch.randn(1, seq_len, input_dim)
    x2 = x1.clone()
    
    # Modify the sequence at index 10
    x2[:, 10:, :] = torch.randn(1, seq_len - 10, input_dim)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Outputs should be perfectly identical before the modification point
    assert torch.allclose(out1[:, :10, :], out2[:, :10, :], atol=1e-6)


# ==========================================
# RNN / GRU / LSTM BACKEND TESTS
# ==========================================

@pytest.mark.parametrize("BackendClass", [RnnBackend, GruBackend, LstmBackend])
def rec_backend_shape_test(BackendClass, dummy_sequence_data, input_dim, hidden_dim):
    """Tests shape for all recurrent backends."""
    model = BackendClass(input_dim, hidden_dim)
    out = model(dummy_sequence_data)
    assert out.shape == (dummy_sequence_data.size(0), dummy_sequence_data.size(1), hidden_dim)

@pytest.mark.parametrize("BackendClass, inner_module_name", [
    (RnnBackend, 'rnn'), 
    (GruBackend, 'gru'), 
    (LstmBackend, 'lstm')
])
def test_recurrent_backend_config(BackendClass, inner_module_name, input_dim, hidden_dim):
    """Tests configuration parsing (dropout behavior for num_layers)."""
    # Multi-layer should have internal dropout
    multi_layer_config = {'num_layers': 3, 'dropout': 0.3}
    model_multi = BackendClass(input_dim, hidden_dim, multi_layer_config)
    inner_module_multi = getattr(model_multi, inner_module_name)
    assert inner_module_multi.num_layers == 3
    assert inner_module_multi.dropout == 0.3
    assert model_multi.output_dropout.p == 0.3

    # Single-layer should force internal dropout to 0.0
    single_layer_config = {'num_layers': 1, 'dropout': 0.5}
    model_single = BackendClass(input_dim, hidden_dim, single_layer_config)
    inner_module_single = getattr(model_single, inner_module_name)
    assert inner_module_single.num_layers == 1
    assert inner_module_single.dropout == 0.0
    assert model_single.output_dropout.p == 0.5


# ==========================================
# CAUSAL TRANSFORMER BACKEND TESTS
# ==========================================

def test_causal_transformer_shape(dummy_sequence_data, input_dim, hidden_dim):
    model = CausalTransformerBackend(input_dim, hidden_dim)
    out = model(dummy_sequence_data)
    assert out.shape == (dummy_sequence_data.size(0), dummy_sequence_data.size(1), hidden_dim)

def test_causal_transformer_causality(input_dim, hidden_dim):
    """Test strict causality of the transformer attention mask."""
    seq_len = 12
    model = CausalTransformerBackend(input_dim, hidden_dim, config={'num_layers': 2, 'nheads': 2})
    model.eval()

    x1 = torch.randn(1, seq_len, input_dim)
    x2 = x1.clone()
    
    # Modify timestep 6
    x2[:, 6:, :] = torch.randn(1, seq_len - 6, input_dim)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # In a causal transformer, changing t=6 should not affect t < 6
    assert torch.allclose(out1[:, :6, :], out2[:, :6, :], atol=1e-5)
    assert not torch.allclose(out1[:, 6:, :], out2[:, 6:, :], atol=1e-5)

# ==========================================
# GATED RESIDUAL BACKEND TESTS
# ==========================================

def test_gated_residual_shape(dummy_sequence_data, input_dim, hidden_dim):
    model = GatedResidualBackend(input_dim, hidden_dim)
    out = model(dummy_sequence_data)
    assert out.shape == (dummy_sequence_data.size(0), dummy_sequence_data.size(1), hidden_dim)

def test_gated_residual_config(input_dim, hidden_dim):
    config = {'dropout': 0.7}
    model = GatedResidualBackend(input_dim, hidden_dim, config)
    
    # Zoek de dropout laag op die genest zit in de nn.Sequential (self.net)
    dropouts = [m for m in model.net.modules() if isinstance(m, torch.nn.Dropout)]
    
    assert len(dropouts) == 1, "Expected exactly 1 dropout layer in GatedResidualBackend"
    assert dropouts[0].p == 0.7, f"Expected dropout p=0.7, got {dropouts[0].p}"