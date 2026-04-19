import pytest
import torch
import copy
import math

from hmoe2.tensor import HmoeTensor
from hmoe2.schema import HmoeFeature
# Pas de import aan naar waar jouw sanitizer leeft
from hmoe2.sanitize import HmoeSanitizerAuto 


@pytest.fixture
def mock_features():
    """Provides a base list of HmoeFeatures."""
    return [
        HmoeFeature(name="go-mtf-delta_5m_14"),
        HmoeFeature(name="go-friction_14"),
        HmoeFeature(name="go-target_50_1.5_1.0"),
        HmoeFeature(name="noise_feature_with_nans"),
        HmoeFeature(name="unauthorized_feature")
    ]

@pytest.fixture
def sample_tensor(mock_features):
    """
    Creates a mock HmoeTensor of shape [Batch(1), Seq(5), Features(5)]
    """
    # F0: Zero-anchored data (mean ~ 0)
    # F1: Median-centered data (mean ~ 100, with one massive outlier)
    # F2: Target data (should never be scaled)
    # F3: Data containing NaNs and Infs
    # F4: Data that should be filtered out
    raw_data = torch.tensor([
        [
            [0.1,  100.0, 1.0, float('nan'), 5.0],
            [-0.1, 101.0, 0.0, 2.0,          5.0],
            [0.2,  99.0,  1.0, 3.0,          5.0],
            [-0.2, 100.5, 0.0, float('inf'), 5.0],
            [0.0,  999.0, 1.0, 5.0,          5.0], # 999.0 is an outlier
        ]
    ], dtype=torch.float32)

    return HmoeTensor(tensor=raw_data, indices=mock_features)


def test_feature_filtering(sample_tensor):
    """Test if unauthorized features are successfully dropped."""
    allowed = [
        HmoeFeature(name="go-mtf-delta_5m_14"),
        HmoeFeature(name="go-friction_14"),
        HmoeFeature(name="go-target_50_1.5_1.0")
    ]
    
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=allowed,
        verbose=False
    )
    
    assert result.tensor.size(-1) == 3, "Output should exactly have 3 features"
    
    result_names = [f.name for f in result.indices]
    assert "unauthorized_feature" not in result_names
    assert "noise_feature_with_nans" not in result_names
    assert "go-mtf-delta_5m_14" in result_names


def test_target_bypass(sample_tensor, mock_features):
    """Test if target/label features are strictly untouched."""
    # Capture original target values
    original_target_slice = sample_tensor.tensor[:, :, 2].clone()
    
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=mock_features, # Allow all for this test
        verbose=False
    )
    
    # Find the index of the target feature in the result
    target_idx = next(i for i, f in enumerate(result.indices) if "target" in f.name.lower())
    result_target_slice = result.tensor[:, :, target_idx]
    
    # Ensure perfect mathematical match
    assert torch.allclose(original_target_slice, result_target_slice), "Target column was mutated!"


def test_nan_handling_replacement(sample_tensor, mock_features):
    """Test if NaNs and Infs are replaced by zeros when drop_nan_columns=False."""
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=mock_features,
        drop_nan_columns=False,
        verbose=False
    )
    
    # Check if there are any NaNs or Infs left
    assert not torch.isnan(result.tensor).any(), "NaNs were not removed"
    assert not torch.isinf(result.tensor).any(), "Infinities were not removed"


def test_nan_handling_drop(sample_tensor, mock_features):
    """Test if columns with NaNs are entirely dropped when drop_nan_columns=True."""
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=mock_features,
        drop_nan_columns=True,
        verbose=False
    )
    
    result_names = [f.name for f in result.indices]
    assert "noise_feature_with_nans" not in result_names, "NaN column was not dropped"


def test_explicit_clamping(sample_tensor):
    """Test if the user-defined active clamps are strictly enforced."""
    allowed = [
        HmoeFeature(name="go-mtf-delta_5m_14", clamp=0.5) # Force strict clamp
    ]
    
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=allowed,
        verbose=False
    )
    
    delta_idx = next(i for i, f in enumerate(result.indices) if f.name == "go-mtf-delta_5m_14")
    delta_slice = result.tensor[:, :, delta_idx]
    
    assert delta_slice.max().item() <= 0.5, f"Clamp failed, max is {delta_slice.max().item()}"
    assert delta_slice.min().item() >= -0.5, f"Clamp failed, min is {delta_slice.min().item()}"


def test_soft_clamping_bounds(sample_tensor):
    """Test if the default [-5.0, 5.0] soft clamp is applied during normalization."""
    allowed = [
        HmoeFeature(name="go-friction_14") # Contains a massive 999.0 outlier
    ]
    
    result = HmoeSanitizerAuto.sanitize(
        raw_tensor=sample_tensor,
        allowed_features=allowed,
        use_robust=False, # Use standard scale so the outlier blows up the score
        verbose=False
    )
    
    # Even without explicit clamp, the standard normalization soft-clips to 5.0
    assert result.tensor.max().item() <= 5.0001, "Soft clip bounds exceeded"
    assert result.tensor.min().item() >= -5.0001, "Soft clip bounds exceeded"


def test_robust_vs_standard_scaling():
    """Test if robust scaling actually suppresses outliers better than standard scaling."""
    # Create a sequence with a stable median but a ridiculous outlier
    raw_data = torch.tensor([[
        [1.0, 1.0], [1.1, 1.1], [0.9, 0.9], [1.0, 1.0], [1000.0, 1000.0]
    ]], dtype=torch.float32)
    
    features = [
        HmoeFeature(name="robust_test"),
        HmoeFeature(name="std_test")
    ]
    
    tensor = HmoeTensor(tensor=raw_data, indices=features)
    
    # Run Robust
    result_robust = HmoeSanitizerAuto.sanitize(
        raw_tensor=tensor, allowed_features=[features[0]], use_robust=True, verbose=False
    )
    
    # Run Standard
    result_std = HmoeSanitizerAuto.sanitize(
        raw_tensor=tensor, allowed_features=[features[1]], use_robust=False, verbose=False
    )
    
    robust_slice = result_robust.tensor[:, :, 0]
    std_slice = result_std.tensor[:, :, 0]
    
    # Robust scaling should keep the non-outlier values relatively close to 0 
    # (since the median is 1.0, subtracting 1.0 makes them ~0.0)
    assert abs(robust_slice[0, 0].item()) < 1.0
    
    # The standard scaler gets its mean destroyed by 1000.0, so the 'normal' values 
    # will be deeply negative (e.g., trying to compensate for the massive mean)
    assert std_slice[0, 0].item() < -0.1 # Meaning it got skewed