from __future__ import annotations
from typing import Union, Any, Dict
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from hmoe2.serializable import Serializable


class HmoeNodeType(Enum):
    """Enumeration of node types within the HMoE architecture.

    This enum defines the two fundamental node categories used
    in the hierarchical mixture-of-experts system.

    Attributes:
        ROUTER: Node responsible for routing inputs to child branches.
        EXPERT: Leaf node responsible for producing task predictions.
    """
    ROUTER = auto()
    EXPERT = auto()


@dataclass(frozen=True)
class HmoeTask(Serializable):
    """Configuration object defining a learning task.

    This class represents a single supervised objective, including
    classification structure, weighting parameters, and label mapping.

    Attributes:
        name (str): Unique identifier for the task.
        num_classes (int): Number of output classes.
        loss_weight (float): Scaling factor for task loss contribution.
        pos_weight (float): Weight applied to positive samples.
        label_target (HmoeFeature): Feature representing ground truth labels.
        enabled (bool): Flag indicating whether the task is active.
    """
    name: str
    num_classes: int
    loss_weight: float = 1.0
    pos_weight: float = 1.0
    label_target: HmoeFeature = None
    enabled: bool = True


@dataclass
class HmoeFeature(Serializable):
    """Feature descriptor used throughout the HMoE pipeline.

    This class defines metadata for a single feature, including optional
    preprocessing directives such as clamping and normalization.

    Attributes:
        name (str): Unique feature identifier.
        clamp (float): Maximum absolute value allowed for this feature.
        normalize (bool): Whether to apply normalization to this feature.
    """
    name: str = None
    clamp: float = 0.0
    normalize: int = 0


@dataclass
class HmoeCheatFeature(HmoeFeature):
    """Special feature type that explicitly flags itself as a cheat feature.

    Cheat features are typically used for debugging, diagnostics, or
    controlled experiments. During serialization, they inject an explicit
    flag to distinguish them from standard features.
    """

    def serialize(self) -> Dict[str, Any]:
        """Serializes the feature while enforcing cheat flag inclusion.

        This method ensures that the serialized representation always
        includes a 'cheat' flag, regardless of the base serialization format.

        Returns:
            Dict[str, Any]: Serialized feature with cheat flag.
        """
        # Perform base serialization using parent implementation
        base_serialization = super().serialize()

        # Ensure serialization is in dictionary form for mutation
        if isinstance(base_serialization, str):
            # Convert string representation into dictionary format
            f_dict = {"name": base_serialization}
        else:
            f_dict = base_serialization

        # Inject cheat flag to explicitly mark feature type
        f_dict["cheat"] = True

        return f_dict