import pytest
import torch

from hmoe2.snn import SurrogateSpike, SnnBackend


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def input_dim():
    return 8

@pytest.fixture
def hidden_dim():
    return 16

@pytest.fixture
def dummy_input(batch_size, seq_len, input_dim):
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, input_dim)


# ==========================================
# SURROGATE SPIKE TESTS (AUTOGRAD)
# ==========================================

def test_surrogate_spike_forward():
    """Test of de harde thresholding correct werkt in de forward pass."""
    threshold = 1.0
    mem = torch.tensor([0.0, 0.99, 1.0, 1.5])
    
    out = SurrogateSpike.apply(mem, threshold)
    
    # Verwachte output: 0 voor < 1.0, 1 voor >= 1.0
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(out, expected), "Forward pass genereert foute spikes."


def test_surrogate_spike_backward_gradient_flow():
    """
    Kritieke test: Test of de surrogate gradient functie daadwerkelijk
    gradients doorlaat waar een normale step-functie 0 of NaN zou geven.
    """
    threshold = 1.0
    
    # Tensor die we willen updaten
    mem = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    
    # Forward pass
    spikes = SurrogateSpike.apply(mem, threshold)
    
    # Dummy loss (som van alle spikes)
    loss = spikes.sum()
    loss.backward()
    
    # Als dit een normale (mem >= threshold).float() was, zou grad 0.0 zijn.
    # Onze surrogate gradient moet echter een vloeiende curve doorgeven.
    assert mem.grad is not None, "Backward pass geeft None gradients."
    
    # De gradient moet groter zijn dan 0 (de sigmoid is overal positief)
    assert (mem.grad > 0).all(), "Gradients worden geblokkeerd door de backward pass."
    
    # De gradient hoort het hoogst te zijn exact OP de threshold
    assert mem.grad[1] > mem.grad[0]
    assert mem.grad[1] > mem.grad[2]


# ==========================================
# SNN BACKEND TESTS
# ==========================================

def test_snn_backend_shape(dummy_input, input_dim, hidden_dim):
    """Test of de SNN backend de juiste output dimensies genereert."""
    model = SnnBackend(input_dim, hidden_dim)
    out = model(dummy_input)
    
    assert out.shape == (dummy_input.size(0), dummy_input.size(1), hidden_dim)


def test_snn_backend_binary_sparse_output(dummy_input, input_dim, hidden_dim):
    """Test of de output van de SNN écht alleen uit 0'en en 1'en bestaat (naast dropout)."""
    # Zet dropout uit voor deze test, anders krijg je geschaalde waardes (zoals 1.25)
    model = SnnBackend(input_dim, hidden_dim, config={'dropout': 0.0})
    model.eval()
    
    out = model(dummy_input)
    
    # Check of alle waardes exact 0.0 of 1.0 zijn
    is_zero = (out == 0.0)
    is_one = (out == 1.0)
    assert (is_zero | is_one).all(), "SNN output bevat niet-binaire (float) waarden!"


def test_snn_backend_dynamics_no_input(input_dim, hidden_dim):
    """Als de input 0 is, mag het netwerk (zonder bias) nooit vuren."""
    model = SnnBackend(input_dim, hidden_dim, config={'dropout': 0.0})
    model.eval()
    
    # Verwijder handmatig de bias uit de MLP lagen voor deze pure test
    for layer in model.fc:
        if isinstance(layer, torch.nn.Linear):
            layer.bias.data.fill_(0.0)
            
    zero_input = torch.zeros(2, 10, input_dim)
    out = model(zero_input)
    
    assert out.sum().item() == 0, "SNN vuurt spikes terwijl er geen input stroom is!"


def test_snn_backend_full_gradient_flow(dummy_input, input_dim, hidden_dim):
    """Test of backpropagation door de gehele tijdsdimensie en SNN integratie slaagt."""
    model = SnnBackend(input_dim, hidden_dim)
    
    # Maak input differentieerbaar
    dummy_input.requires_grad_(True)
    
    out = model(dummy_input)
    loss = out.sum()
    loss.backward()
    
    assert dummy_input.grad is not None, "Gradients stromen niet terug naar de input."
    assert not torch.isnan(dummy_input.grad).any(), "NaN gradients gedetecteerd in SNN."
    assert dummy_input.grad.abs().sum() > 0, "Gradients naar input zijn volledig nul."


def test_snn_strict_causality(input_dim, hidden_dim):
    """
    Test wiskundige causaliteit: een wijziging in de toekomst 
    mag NOOIT een spike in het verleden beïnvloeden.
    """
    seq_len = 20
    model = SnnBackend(input_dim, hidden_dim, config={'dropout': 0.0})
    model.eval()

    x1 = torch.randn(1, seq_len, input_dim)
    x2 = x1.clone()
    
    # Verander het signaal vanaf tijdstap 10
    x2[:, 10:, :] = torch.randn(1, seq_len - 10, input_dim)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Spikes voor t=0 tot t=9 moeten 100% identiek zijn
    assert torch.allclose(out1[:, :10, :], out2[:, :10, :]), \
        "Causaliteitslek: Toekomstige stroom heeft spikes in het verleden veranderd."