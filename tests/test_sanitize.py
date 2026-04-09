import pytest
import torch
import logging
from typing import List

from hmoe2.sanitize import HmoeSanitizer
from hmoe2.tensor import HmoeTensor
from hmoe2.schema import HmoeFeature

# ==========================================
# MOCKS AND FIXTURES
# ==========================================

# Minimal mock for HmoeFeature to isolate tests
class MockHmoeFeature:
    def __init__(self, name: str, clamp: float = 0.0, normalize: int = 0):
        self.name = name
        self.clamp = clamp
        self.normalize = normalize

# Minimal mock for HmoeTensor
class MockHmoeTensor:
    def __init__(self, tensor: torch.Tensor, indices: List[MockHmoeFeature]):
        self.tensor = tensor
        self.indices = indices

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def base_features():
    return [
        MockHmoeFeature("price"),
        MockHmoeFeature("volume"),
        MockHmoeFeature("rsi")
    ]

@pytest.fixture
def dummy_raw_tensor(batch_size, seq_len, base_features):
    # Shape: [Batch=2, Seq=10, Features=3]
    tensor_data = torch.randn(batch_size, seq_len, len(base_features))
    return MockHmoeTensor(tensor=tensor_data, indices=base_features)

# ==========================================
# FILTERING TESTS
# ==========================================

def test_feature_whitelisting(dummy_raw_tensor):
    """Validates that unapproved features are filtered out."""
    # We only allow 'price' and 'rsi'. 'volume' should be dropped.
    allowed = [MockHmoeFeature("price"), MockHmoeFeature("rsi")]
    
    sanitized = HmoeSanitizer.sanitize(dummy_raw_tensor, allowed_features=allowed, verbose=False)
    
    # Check tensor shape (1 feature dropped)
    assert sanitized.tensor.shape == (2, 10, 2)
    
    # Check updated indices
    assert len(sanitized.indices) == 2
    assert sanitized.indices[0].name == "price"
    assert sanitized.indices[1].name == "rsi"

def test_feature_wildcard_matching(dummy_raw_tensor):
    """Validates that wildcard patterns (e.g., matching a prefix) work."""
    # Inject a feature with a sub-prefix
    dummy_raw_tensor.indices.append(MockHmoeFeature("price__lag1"))
    dummy_raw_tensor.tensor = torch.cat([dummy_raw_tensor.tensor, torch.randn(2, 10, 1)], dim=2)
    
    # Allow 'price'. This should also catch 'price__lag1'
    allowed = [MockHmoeFeature("price")]
    
    sanitized = HmoeSanitizer.sanitize(dummy_raw_tensor, allowed_features=allowed, verbose=False)
    
    assert sanitized.tensor.shape == (2, 10, 2)
    assert sanitized.indices[0].name == "price"
    assert sanitized.indices[1].name == "price__lag1"

# ==========================================
# NaN HANDLING TESTS
# ==========================================

def test_nan_imputation(dummy_raw_tensor):
    """Validates that NaNs are replaced with zeros when drop_nan_columns=False."""
    # Inject NaNs into the volume column (index 1)
    dummy_raw_tensor.tensor[:, :, 1] = float('nan')
    
    allowed = dummy_raw_tensor.indices # Allow all
    
    sanitized = HmoeSanitizer.sanitize(
        dummy_raw_tensor, allowed_features=allowed, drop_nan_columns=False, verbose=False
    )
    
    # The column should still exist
    assert sanitized.tensor.shape == (2, 10, 3)
    
    # The NaNs should have become zeros
    assert torch.all(sanitized.tensor[:, :, 1] == 0.0)
    assert not torch.isnan(sanitized.tensor).any()

def test_drop_nan_columns(dummy_raw_tensor):
    """Validates that entire columns are dropped if drop_nan_columns=True."""
    # Inject a NaN into the volume column (index 1)
    dummy_raw_tensor.tensor[0, 5, 1] = float('nan')
    
    allowed = dummy_raw_tensor.indices
    
    sanitized = HmoeSanitizer.sanitize(
        dummy_raw_tensor, allowed_features=allowed, drop_nan_columns=True, verbose=False
    )
    
    # The column containing the NaN should be completely removed
    assert sanitized.tensor.shape == (2, 10, 2)
    assert len(sanitized.indices) == 2
    assert sanitized.indices[0].name == "price"
    assert sanitized.indices[1].name == "rsi"

# ==========================================
# NORMALIZATION TESTS
# ==========================================

def test_static_normalization_type_1(dummy_raw_tensor):
    """Validates static scaling (/ 100)."""
    # Force RSI data to be 50.0
    dummy_raw_tensor.tensor[:, :, 2] = 50.0
    
    # Configure RSI to use Type 1 normalization
    allowed = [MockHmoeFeature("rsi", normalize=1)]
    
    sanitized = HmoeSanitizer.sanitize(dummy_raw_tensor, allowed_features=allowed, verbose=False)
    
    # Data should be 50.0 / 100.0 = 0.5
    assert torch.allclose(sanitized.tensor[:, :, 0], torch.tensor(0.5))

def test_rolling_normalization_type_2(dummy_raw_tensor):
    """Validates Rolling Z-Score normalization."""
    # Create an ascending trend for the price column to test rolling stats
    trend = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, 10)
    dummy_raw_tensor.tensor[:, :, 0] = trend
    
    # Configure Price to use Type 2 normalization (Rolling Z-Score) with a small window
    allowed = [MockHmoeFeature("price", normalize=2)]
    
    sanitized = HmoeSanitizer.sanitize(
        dummy_raw_tensor, allowed_features=allowed, rolling_window=3, verbose=False
    )
    
    result = sanitized.tensor[:, :, 0]
    
    # Math Verification for Batch 0:
    # Sequence = [0, 1, 2, 3, 4, ...]
    # Window = 3
    # At t=2 (vals: [0, 1, 2]): mean = 1.0, std = 0.8165
    # Normalized t=2 = (2 - 1.0) / 0.8165 = 1.2247
    
    # Allow a small tolerance for floating point cumulative sum math
    expected_val_t2 = (2.0 - 1.0) / torch.tensor([0.0, 1.0, 2.0]).std(unbiased=False)
    assert torch.allclose(result[0, 2], expected_val_t2, atol=1e-4)

# ==========================================
# CLAMPING TESTS
# ==========================================

def test_value_clamping(dummy_raw_tensor):
    """Validates that values exceeding boundaries are hard-capped."""
    # Inject extreme values into the 'price' column (index 0)
    dummy_raw_tensor.tensor[0, 0, 0] = 500.0
    dummy_raw_tensor.tensor[0, 1, 0] = -500.0
    
    # Allow 'price' and set a hard clamp of 5.0
    allowed = [MockHmoeFeature("price", clamp=5.0)]
    
    sanitized = HmoeSanitizer.sanitize(dummy_raw_tensor, allowed_features=allowed, verbose=False)
    
    # Extract the sanitized price column
    price_col = sanitized.tensor[:, :, 0]
    
    # Values should not exceed +/- 5.0
    assert torch.max(price_col) <= 5.0
    assert torch.min(price_col) >= -5.0
    
    # The extreme values should specifically equal the clamp boundaries
    assert price_col[0, 0] == 5.0
    assert price_col[0, 1] == -5.0