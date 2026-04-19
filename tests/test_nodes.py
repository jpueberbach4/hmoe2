import pytest
import yaml
from unittest.mock import MagicMock, patch, mock_open

from hmoe2.nodes import HmoeNode, parse_feature, parse_task
from hmoe2.schema import HmoeFeature, HmoeCheatFeature, HmoeTask, HmoeNodeType

# ==========================================
# MOCKS AND FIXTURES
# ==========================================

class DummyNode(HmoeNode):
    """A concrete implementation of HmoeNode strictly for testing abstract instance methods."""
    def __init__(self, *args, **kwargs):
        super(HmoeNode, self).__init__() # Initialize the nn.Module internals
        # Map the kwargs to attributes manually for the dataclass behavior
        for k, v in kwargs.items():
            setattr(self, k, v)
    def forward(self, payload):
        pass
    
    def _serialize_node(self):
        return {"name": self.name, "type": self.type.name if self.type else "DUMMY"}
    
    def _gather_tasks(self, task_dict):
        # Insert a fake task to test aggregation
        dummy_task = MagicMock()
        dummy_task.serialize.return_value = {"name": "dummy_gathered_task"}
        task_dict["dummy_gathered_task"] = dummy_task


@pytest.fixture
def sample_yaml_string():
    return """
    tasks:
      - name: "price_prediction"
        num_classes: 3
        loss_weight: 1.5
        label_target: "future_price"
    tree:
      name: "root_router"
      type: "ROUTER"
      children:
        - name: "finance_expert"
          type: "EXPERT"
          allowed_tasks: ["price_prediction"]
          features:
            - name: "volume"
            - name: "price"
              cheat: true
    """

@pytest.fixture
def sample_dict_config(sample_yaml_string):
    return yaml.safe_load(sample_yaml_string)


# ==========================================
# PARSER TESTS
# ==========================================

def test_parse_feature_string():
    """Validates simple string feature parsing."""
    feat = parse_feature("simple_feature")
    # Assuming HmoeFeature has a 'name' attribute
    assert feat.name == "simple_feature"
    assert type(feat).__name__ == "HmoeFeature"

def test_parse_feature_dict_standard():
    """Validates dictionary parsing for standard features."""
    f_data = {"name": "norm_feat", "clamp": 5.0, "normalize": 1}
    feat = parse_feature(f_data)
    
    assert feat.name == "norm_feat"
    assert feat.clamp == 5.0
    assert feat.normalize == 1
    assert type(feat).__name__ == "HmoeFeature"

def test_parse_feature_dict_cheat():
    """Validates dictionary parsing properly routes to HmoeCheatFeature."""
    f_data = {"name": "future_leak", "cheat": True}
    feat = parse_feature(f_data)
    
    assert feat.name == "future_leak"
    assert type(feat).__name__ == "HmoeCheatFeature"

def test_parse_task_basic():
    """Validates task definition parsing."""
    t_data = {"name": "basic_task", "num_classes": 2}
    task = parse_task(t_data)
    
    assert task.name == "basic_task"
    assert task.num_classes == 2
    assert task.loss_weight == 1.0 # Default fallback
    assert task.enabled is True    # Default fallback
    assert task.label_target is None

def test_parse_task_with_label_target():
    """Validates nested feature parsing within task parsing."""
    t_data = {
        "name": "complex_task", 
        "num_classes": 5, 
        "label_target": {"name": "target_feature", "cheat": True}
    }
    task = parse_task(t_data)
    
    assert task.label_target is not None
    assert type(task.label_target).__name__ == "HmoeCheatFeature"
    assert task.label_target.name == "target_feature"


# ==========================================
# FILE I/O SERIALIZATION TESTS
# ==========================================

@patch('builtins.open', new_callable=mock_open, read_data="tasks: []\ntree: {}")
@patch.object(HmoeNode, 'from_dict')
def test_from_yaml(mock_from_dict, mock_file):
    """Validates reading from YAML properly delegates to from_dict."""
    # Since HmoeNode is an ABC, we call from_yaml on the DummyNode or patch the ABC
    # The classmethod doesn't rely on self, so calling it on HmoeNode works if from_dict is patched
    HmoeNode.from_yaml("dummy/path.yaml")
    
    mock_file.assert_called_once_with("dummy/path.yaml", 'r')
    mock_from_dict.assert_called_once_with({'tasks': [], 'tree': {}})

@patch('builtins.open', new_callable=mock_open)
@patch('yaml.dump')
def test_to_yaml(mock_yaml_dump, mock_file):
    """Validates writing to YAML translates the internal dict representation properly."""
    node = DummyNode(name="test_node")
    
    # We patch to_dict on the instance so we control the output passed to yaml.dump
    with patch.object(node, 'to_dict', return_value={"mock": "data"}):
        node.to_yaml("output/path.yaml")
        
        mock_file.assert_called_once_with("output/path.yaml", 'w')
        # Check that yaml.dump was called with the dict and the file handle
        call_args, call_kwargs = mock_yaml_dump.call_args
        assert call_args[0] == {"mock": "data"}
        assert call_kwargs["default_flow_style"] is False


# ==========================================
# DICTIONARY SERIALIZATION TESTS
# ==========================================

@patch('hmoe2.nodes.parse_task')
@patch.object(HmoeNode, '_build_node')
def test_from_dict(mock_build_node, mock_parse_task):
    """Validates root dictionary parsing and delegation to _build_node."""
    mock_task = MagicMock()
    mock_parse_task.return_value = mock_task
    
    data_dict = {
        'tasks': [{'name': 't1', 'num_classes': 2}],
        'tree': {'type': 'ROUTER', 'name': 'root'}
    }
    
    HmoeNode.from_dict(data_dict)
    
    mock_parse_task.assert_called_once()
    mock_build_node.assert_called_once_with({'type': 'ROUTER', 'name': 'root'}, [mock_task])

def test_to_dict():
    """Validates the structure of the serialized dictionary from a concrete node."""
    node = DummyNode(name="test_node", type=MagicMock(name="DUMMY"))
    
    result = node.to_dict()
    
    # Check that tasks were gathered via the abstract method implementation
    assert 'tasks' in result
    assert len(result['tasks']) == 1
    assert result['tasks'][0] == {"name": "dummy_gathered_task"}
    
    # Check tree serialization via the abstract method implementation
    assert 'tree' in result
    assert result['tree']['name'] == "test_node"


# ==========================================
# DYNAMIC NODE BUILDING TESTS
# ==========================================

@patch('hmoe2.routers.HmoeRouter', create=True)
@patch('hmoe2.experts.HmoeExpert', create=True)
def test_build_node_router(MockExpert, MockRouter):
    """Validates recursive construction of a ROUTER and its children."""
    # Setup the mock router instance behavior
    mock_router_instance = MagicMock()
    mock_router_instance.branches = []
    MockRouter.return_value = mock_router_instance
    
    config = {
        'type': 'ROUTER',
        'name': 'root',
        'children': [{'type': 'ROUTER', 'name': 'child1'}]
    }
    
    node = HmoeNode._build_node(config, global_tasks=[])
    
    # The router should have been instantiated twice (root and child)
    assert MockRouter.call_count == 2
    MockRouter.assert_any_call(name='root', config=config)
    
    # Gate should be built after children are attached
    assert mock_router_instance.build_gate.call_count == 2
    assert node == mock_router_instance

@patch('hmoe2.experts.HmoeExpert', create=True)
@patch('hmoe2.nodes.parse_feature')
def test_build_node_expert(mock_parse_feature, MockExpert):
    """Validates specific construction logic for an EXPERT node."""
    mock_expert_instance = MagicMock()
    MockExpert.return_value = mock_expert_instance
    
    mock_feature = MagicMock()
    mock_parse_feature.return_value = mock_feature
    
    global_task_a = MagicMock()
    global_task_a.name = "task_a"
    global_task_b = MagicMock()
    global_task_b.name = "task_b"
    
    config = {
        'type': 'EXPERT',
        'name': 'specialist',
        'allowed_tasks': ['task_b'], # Should filter out task_a
        'features': ['feat1'],
        'backend': 'TCN',
        'hidden_dim': 64
    }
    
    node = HmoeNode._build_node(config, global_tasks=[global_task_a, global_task_b])
    
    # Validate expert instantiation with correct parsed configs
    MockExpert.assert_called_once_with(
        name='specialist',
        tasks=[global_task_b], # filtered correctly
        features=[mock_feature],
        backend='TCN',
        hidden_dim=64,
        config=config
    )
    
    # Validate task linking
    mock_expert_instance.link_tasks.assert_called_once_with([global_task_b])
    assert node == mock_expert_instance

def test_build_node_unknown_type():
    """Validates that corrupt YAML types trigger a clear ValueError."""
    config = {'type': 'GHOST', 'name': 'phantom_node'}
    with pytest.raises(ValueError, match="Unknown node type: GHOST"):
        HmoeNode._build_node(config, global_tasks=[])


# ==========================================
# SUBTREE FEATURE AGGREGATION TESTS
# ==========================================

def test_subtree_features_expert():
    """An expert should simply return its own local feature list."""
    f1 = MagicMock()
    f1.name = "a"
    node = DummyNode(type=MagicMock(value=2)) # Mimic HmoeNodeType.EXPERT
    # Patch the enum check logic
    with patch('hmoe2.nodes.HmoeNodeType') as mock_enum:
        node.type = mock_enum.EXPERT
        node.features = [f1]
        
        features = node.subtree_features
        assert features == [f1]

def test_subtree_features_router():
    """A router should recurse into branches, aggregate unique features, and sort them.
    It must also include its own local features."""
    
    # Features embedded inside the child branches
    f_z = MagicMock(); f_z.name = "z"
    f_a = MagicMock(); f_a.name = "a"
    f_dup = MagicMock(); f_dup.name = "a" # Duplicate feature in another branch
    
    # Local feature defined strictly on the Router itself
    f_local = MagicMock(); f_local.name = "router_local_feat"
    
    # Setup mock children with pre-defined subtree_features
    child1 = MagicMock()
    child1.subtree_features = [f_z, f_a]
    
    child2 = MagicMock()
    child2.subtree_features = [f_dup]
    
    # Instantiate DummyNode and explicitly set the new 'features' attribute
    node = DummyNode(features=[f_local])
    node.branches = [child1, child2]
    
    with patch('hmoe2.nodes.HmoeNodeType') as mock_enum:
        node.type = mock_enum.ROUTER
        
        features = node.subtree_features
        
        # Should deduplicate "a", include the local feature, and sort alphabetically: 
        # ["a", "router_local_feat", "z"]
        assert len(features) == 3
        assert features[0].name == "a"
        assert features[1].name == "router_local_feat"
        assert features[2].name == "z"