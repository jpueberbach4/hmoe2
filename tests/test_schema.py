import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from hmoe2.schema import HmoeNodeType, HmoeTask, HmoeFeature, HmoeCheatFeature

# ==========================================
# HMOE NODE TYPE (ENUM) TESTS
# ==========================================

def test_hmoe_node_type_members():
    """Validates that the enum contains exactly the expected routing boundaries."""
    assert hasattr(HmoeNodeType, 'ROUTER')
    assert hasattr(HmoeNodeType, 'EXPERT')
    
    # Ensure they are uniquely auto-assigned
    assert HmoeNodeType.ROUTER != HmoeNodeType.EXPERT
    assert HmoeNodeType.ROUTER.value != HmoeNodeType.EXPERT.value


# ==========================================
# HMOE FEATURE TESTS
# ==========================================

def test_hmoe_feature_defaults():
    """Validates the fallback values for standard features."""
    feat = HmoeFeature(name="raw_price")
    
    assert feat.name == "raw_price"
    assert feat.clamp == 0.0
    assert feat.normalize == 0

def test_hmoe_feature_initialization():
    """Validates that feature attributes are properly assigned."""
    feat = HmoeFeature(name="volume", clamp=10.0, normalize=2)
    
    assert feat.name == "volume"
    assert feat.clamp == 10.0
    assert feat.normalize == 2


# ==========================================
# HMOE TASK TESTS
# ==========================================

def test_hmoe_task_defaults():
    """Validates initialization and default weighting of tasks."""
    task = HmoeTask(name="direction_pred", num_classes=3)
    
    assert task.name == "direction_pred"
    assert task.num_classes == 3
    assert task.loss_weight == 1.0
    assert task.pos_weight == 1.0
    assert task.label_target is None
    assert task.enabled is True

def test_hmoe_task_with_label_target():
    """Validates nesting a feature descriptor inside the task."""
    target_feat = HmoeFeature(name="future_return")
    task = HmoeTask(name="return_pred", num_classes=1, label_target=target_feat)
    
    assert task.label_target.name == "future_return"
    assert isinstance(task.label_target, HmoeFeature)

def test_hmoe_task_is_frozen():
    """
    Validates immutability. Since HmoeTask is defined as frozen=True,
    modifying it post-instantiation must throw a FrozenInstanceError.
    """
    task = HmoeTask(name="frozen_task", num_classes=2)
    
    with pytest.raises(FrozenInstanceError):
        task.loss_weight = 2.0
        
    with pytest.raises(FrozenInstanceError):
        task.enabled = False


# ==========================================
# HMOE CHEAT FEATURE TESTS
# ==========================================

def test_cheat_feature_inheritance():
    """Validates that a cheat feature retains standard feature attributes."""
    cheat = HmoeCheatFeature(name="leak_data", clamp=5.0)
    
    assert isinstance(cheat, HmoeFeature)
    assert cheat.name == "leak_data"
    assert cheat.clamp == 5.0

@patch('hmoe2.schema.HmoeFeature.serialize')
def test_cheat_feature_serialize_from_string(mock_super_serialize):
    """
    Validates the override logic when the base serialization returns a raw string.
    The string should be wrapped into a dict, and the cheat flag injected.
    """
    # Mock the Serializable parent class returning just the name
    mock_super_serialize.return_value = "simple_feature_name"
    
    cheat = HmoeCheatFeature(name="simple_feature_name")
    serialized = cheat.serialize()
    
    # Assert base serialization was triggered
    mock_super_serialize.assert_called_once()
    
    # Assert string conversion and flag injection
    assert isinstance(serialized, dict)
    assert serialized["name"] == "simple_feature_name"
    assert serialized["cheat"] is True

@patch('hmoe2.schema.HmoeFeature.serialize')
def test_cheat_feature_serialize_from_dict(mock_super_serialize):
    """
    Validates the override logic when the base serialization returns a dictionary.
    The existing dictionary should be mutated to include the cheat flag.
    """
    # Mock the Serializable parent class returning a full dictionary
    mock_super_serialize.return_value = {
        "name": "complex_feature",
        "clamp": 10.0,
        "normalize": 1
    }
    
    cheat = HmoeCheatFeature(name="complex_feature", clamp=10.0, normalize=1)
    serialized = cheat.serialize()
    
    mock_super_serialize.assert_called_once()
    
    # Assert the dictionary was retained and the flag injected
    assert serialized["name"] == "complex_feature"
    assert serialized["clamp"] == 10.0
    assert serialized["normalize"] == 1
    assert serialized["cheat"] is True