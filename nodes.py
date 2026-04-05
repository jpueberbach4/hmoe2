from __future__ import annotations
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Dict
import yaml

from hmoe2.schema import HmoeCheatFeature, HmoeFeature, HmoeTask, HmoeNodeType
from hmoe2.tensor import HmoeInput, HmoeOutput


def parse_feature(f_data: Any) -> HmoeFeature:
    """Parses a feature definition from YAML-compatible input.

    This function supports both string-based and dictionary-based feature
    specifications, converting them into appropriate HmoeFeature or
    HmoeCheatFeature instances.

    Args:
        f_data (Any): Raw feature definition from YAML (string or dict).

    Returns:
        HmoeFeature: Parsed feature object.
    """
    # Handle simple string-based feature definitions
    if isinstance(f_data, str):
        return HmoeFeature(name=f_data)

    # Extract feature properties from dictionary input
    f_name = f_data.get('name')
    f_clamp = f_data.get('clamp', 0.0)
    f_norm = f_data.get('normalize', False)

    # Determine if feature should be treated as a cheat feature
    if f_data.get('cheat', False):
        return HmoeCheatFeature(name=f_name, clamp=f_clamp, normalize=f_norm)

    # Return standard feature instance
    return HmoeFeature(name=f_name, clamp=f_clamp, normalize=f_norm)


def parse_task(t_data: Dict[str, Any]) -> HmoeTask:
    """Parses a task definition from a dictionary.

    Converts raw dictionary input into a structured HmoeTask object,
    including parsing of label target features if present.

    Args:
        t_data (Dict[str, Any]): Dictionary containing task configuration.

    Returns:
        HmoeTask: Parsed task object.
    """
    # Extract and parse label target feature if provided
    raw_label = t_data.get('label_target')
    parsed_label = parse_feature(raw_label) if raw_label else None

    # Construct and return task configuration object
    return HmoeTask(
        name=t_data['name'],
        num_classes=t_data['num_classes'],
        loss_weight=t_data.get('loss_weight', 1.0),
        pos_weight=t_data.get('pos_weight', 1.0),
        enabled=t_data.get('enabled', True),
        label_target=parsed_label
    )


@dataclass(eq=False)
class HmoeNode(nn.Module, ABC):
    """Abstract base class for all nodes in the HMoE tree.

    This class defines the core interface and serialization logic for
    hierarchical Mixture-of-Experts nodes, including routers and experts.

    Attributes:
        name (str): Unique identifier for the node.
        type (HmoeNodeType): Type of node (e.g., ROUTER or EXPERT).
        features (List[HmoeFeature]): Features associated with the node.
    """

    name: str = None
    type: HmoeNodeType = None
    features: List[HmoeFeature] = field(default_factory=list)

    @abstractmethod
    def forward(self, payload: HmoeInput) -> HmoeOutput:
        """Performs forward computation for the node.

        Args:
            payload (HmoeInput): Input payload containing feature data.

        Returns:
            HmoeOutput: Output predictions or routing results.
        """
        pass

    @classmethod
    def from_yaml(cls, filepath: str) -> 'HmoeNode':
        """Loads a node tree from a YAML file.

        Args:
            filepath (str): Path to the YAML configuration file.

        Returns:
            HmoeNode: Root node of the constructed tree.
        """
        # Read YAML file and parse contents into a dictionary
        with open(filepath, 'r') as f:
            parsed_yaml = yaml.safe_load(f)

        # Build node tree from parsed dictionary
        return cls.from_dict(parsed_yaml)

    def to_yaml(self, filepath: str) -> None:
        """Serializes the node tree to a YAML file.

        Args:
            filepath (str): Destination file path.
        """
        # Convert node tree to dictionary and write to YAML
        with open(filepath, 'w') as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False
            )

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'HmoeNode':
        """Constructs a node tree from a dictionary.

        Args:
            data_dict (Dict[str, Any]): Dictionary containing tasks and tree structure.

        Returns:
            HmoeNode: Root node of the constructed tree.
        """
        # Parse global task configurations
        global_tasks = [parse_task(t) for t in data_dict.get('tasks', [])]

        # Extract tree configuration
        tree_config = data_dict.get('tree', {})

        # Recursively build node hierarchy
        return cls._build_node(tree_config, global_tasks)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node tree into a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the tree.
        """
        # Collect all unique tasks across the subtree
        unique_tasks: Dict[str, HmoeTask] = {}
        self._gather_tasks(unique_tasks)

        # Return serialized tasks and tree structure
        return {
            'tasks': [task.serialize() for task in unique_tasks.values()],
            'tree': self._serialize_node()
        }

    @classmethod
    def _build_node(cls, config: Dict[str, Any], global_tasks: List[HmoeTask]) -> 'HmoeNode':
        """Recursively constructs nodes from configuration.

        Args:
            config (Dict[str, Any]): Node configuration dictionary.
            global_tasks (List[HmoeTask]): List of globally defined tasks.

        Returns:
            HmoeNode: Constructed node instance.

        Raises:
            ValueError: If node type is unknown.
        """
        # Extract node type and name
        node_type = config.get('type', '').upper()
        name = config.get('name', 'unnamed')

        # Handle router node construction
        if node_type == 'ROUTER':
            router = HmoeRouter(name=name, config=config)

            # Recursively build child nodes
            child_configs = config.get('children', [])
            for c in child_configs:
                router.branches.append(cls._build_node(c, global_tasks))

            # Initialize routing gate after children are attached
            router.build_gate()
            return router

        # Handle expert node construction
        elif node_type == 'EXPERT':
            # Determine which tasks are assigned to this expert
            allowed_task_names = config.get('allowed_tasks', None)
            if allowed_task_names is not None:
                expert_tasks = [t for t in global_tasks if t.name in allowed_task_names]
            else:
                expert_tasks = global_tasks

            # Parse feature definitions for this expert
            parsed_features = [parse_feature(f) for f in config.get('features', [])]

            # Extract backend configuration parameters
            backend_val = config.get('backend', 'LINEAR')
            hidden_dim_val = config.get('hidden_dim', 32)
            dilations_val = config.get('dilations', None)

            # Instantiate expert node
            expert = HmoeExpert(
                name=name,
                tasks=expert_tasks,
                features=parsed_features,
                backend=backend_val,
                hidden_dim=hidden_dim_val,
                dilations=dilations_val
            )

            # Link task heads to expert
            expert.link_tasks(expert_tasks)
            return expert

        # Raise error for unknown node types
        else:
            raise ValueError(f"Unknown node type: {node_type} in config for '{name}'")

    @abstractmethod
    def _serialize_node(self) -> Dict[str, Any]:
        """Serializes node-specific configuration.

        Returns:
            Dict[str, Any]: Serialized node data.
        """
        pass

    @abstractmethod
    def _gather_tasks(self, task_dict: Dict[str, HmoeTask]) -> None:
        """Collects tasks from the subtree.

        Args:
            task_dict (Dict[str, HmoeTask]): Dictionary to populate with tasks.
        """
        pass

    @property
    def subtree_features(self) -> List[HmoeFeature]:
        """Computes the set of features used across the subtree.

        Returns:
            List[HmoeFeature]: Sorted list of unique features.
        """
        # If node is an expert, return its local features
        if self.type == HmoeNodeType.EXPERT:
            return self.features

        # Aggregate features from all child nodes
        feature_dict = {}

        if self.type == HmoeNodeType.ROUTER:
            for child in getattr(self, 'branches', []):
                for f in child.subtree_features:
                    if f.name not in feature_dict:
                        feature_dict[f.name] = f

        # Return features sorted by name for consistency
        return sorted(feature_dict.values(), key=lambda f: f.name)


from hmoe2.routers import HmoeRouter
from hmoe2.experts import HmoeExpert