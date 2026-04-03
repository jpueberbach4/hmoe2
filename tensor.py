from __future__ import annotations
import torch
from typing import List, Dict
from dataclasses import dataclass, field
from hmoe2.schema import HmoeFeature


@dataclass(frozen=True)
class HmoeTensor:
    """Immutable tensor wrapper with feature index tracking.

    This class encapsulates a PyTorch tensor along with its corresponding
    feature metadata, enabling safe slicing, device transfers, and
    schema-aware operations.

    Attributes:
        tensor (torch.Tensor): Underlying numerical data.
        indices (List[HmoeFeature]): Feature metadata describing tensor columns.
    """
    tensor: torch.Tensor
    indices: List[HmoeFeature] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, list]) -> HmoeTensor:
        """Constructs an HmoeTensor from a dictionary of raw feature data.

        Each key in the dictionary represents a feature name, and each value
        is a list of raw values. Values are coerced into float format, with
        invalid entries converted to NaN before final sanitization.

        Args:
            data_dict (Dict[str, list]): Mapping of feature names to raw values.

        Returns:
            HmoeTensor: Structured tensor with cleaned numerical data.
        """
        # Initialize containers for feature metadata and cleaned data
        features: List[HmoeFeature] = []
        cleaned_columns: List[List[float]] = []

        # Iterate through input dictionary to build feature list and clean values
        for feature_name, raw_values in data_dict.items():
            # Create feature object for each key
            features.append(HmoeFeature(name=feature_name))

            clean_col = []

            # Convert each value to float, handling invalid entries
            for val in raw_values:
                try:
                    clean_col.append(float(val))
                except (ValueError, TypeError):
                    # Replace invalid values with NaN
                    clean_col.append(float('nan'))

            # Append cleaned column data
            cleaned_columns.append(clean_col)

        # Convert list of columns into tensor format
        tensor_data = torch.tensor(cleaned_columns, dtype=torch.float32)

        # Transpose to shape [sequence, features]
        tensor_data = tensor_data.transpose(0, 1)

        # Replace NaN and infinite values to ensure numerical stability
        tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure tensor has 3 dimensions [batch, sequence, features]
        if tensor_data.dim() == 2:
            tensor_data = tensor_data.unsqueeze(0)

        # Return structured tensor with associated feature metadata
        return cls(tensor=tensor_data, indices=features)

    def get_indices(self) -> List[HmoeFeature]:
        """Returns the feature indices associated with the tensor.

        Returns:
            List[HmoeFeature]: Feature metadata list.
        """
        return self.indices

    def get_subset(self, requested_features: List[HmoeFeature]) -> HmoeTensor:
        """Extracts a subset of features based on requested feature names.

        Feature matching is performed strictly by name to avoid issues with
        object identity or dataclass equality mismatches.

        Args:
            requested_features (List[HmoeFeature]): Features to extract.

        Returns:
            HmoeTensor: New tensor containing only requested features.

        Raises:
            ValueError: If a requested feature is not present.
        """
        # Extract available feature names from current indices
        available_names = [f.name for f in self.indices]

        try:
            # Map requested feature names to their corresponding indices
            raw_integers = [available_names.index(req_f.name) for req_f in requested_features]
        except ValueError:
            # Raise explicit error if feature lookup fails
            raise ValueError(
                "Feature translation failed. Requested feature not found in this tensor."
            )

        # Slice tensor along feature dimension
        sliced_data = self.tensor[:, :, raw_integers]

        # Return new HmoeTensor with sliced data and updated indices
        return HmoeTensor(
            tensor=sliced_data,
            indices=requested_features
        )

    def to(self, device: torch.device) -> 'HmoeTensor':
        """Moves tensor data to the specified device.

        Args:
            device (torch.device): Target device (CPU or GPU).

        Returns:
            HmoeTensor: New tensor instance on target device.
        """
        return HmoeTensor(
            tensor=self.tensor.to(device),
            indices=self.indices
        )

    def to_tensor(self) -> torch.Tensor:
        """Returns the underlying PyTorch tensor.

        Returns:
            torch.Tensor: Raw tensor data.
        """
        return self.tensor


@dataclass(frozen=True)
class HmoeInput:
    """Wrapper for model input payloads.

    This class ensures that input data remains structured and compatible
    with downstream nodes while providing utility methods for slicing
    and device movement.

    Attributes:
        tensor (HmoeTensor): Underlying tensor with feature metadata.
    """
    tensor: HmoeTensor

    def get_subset(self, requested_features: List[HmoeFeature]) -> 'HmoeInput':
        """Extracts a subset of features from the input.

        Delegates slicing to the underlying HmoeTensor and returns
        a new HmoeInput instance.

        Args:
            requested_features (List[HmoeFeature]): Features to extract.

        Returns:
            HmoeInput: New input containing only requested features.
        """
        # Slice underlying tensor
        sliced_tensor = self.tensor.get_subset(requested_features)

        # Wrap result back into HmoeInput
        return HmoeInput(tensor=sliced_tensor)

    def to(self, device: torch.device) -> 'HmoeInput':
        """Moves input tensor to the specified device.

        Args:
            device (torch.device): Target device.

        Returns:
            HmoeInput: New input instance on target device.
        """
        return HmoeInput(tensor=self.tensor.to(device))

    def to_tensor(self) -> torch.Tensor:
        """Extracts raw tensor data from the input.

        Returns:
            torch.Tensor: Underlying numerical tensor.
        """
        return self.tensor.to_tensor()


@dataclass(frozen=True)
class HmoeOutput:
    """Container for model outputs and routing penalties.

    This class encapsulates predictions from multiple task heads along
    with an optional routing loss used for regularization.

    Attributes:
        task_logits (Dict[str, HmoeTensor]): Mapping of task names to predictions.
        routing_loss (HmoeTensor): Aggregated routing penalty.
    """
    task_logits: Dict[str, HmoeTensor]

    routing_loss: HmoeTensor = field(
        default_factory=lambda: HmoeTensor(tensor=torch.tensor(0.0), indices=[])
    )

    def to(self, device: torch.device) -> 'HmoeOutput':
        """Moves all contained tensors to the specified device.

        This includes both task logits and routing loss.

        Args:
            device (torch.device): Target device.

        Returns:
            HmoeOutput: New output instance with tensors on target device.
        """
        # Move each task tensor individually
        moved_logits = {k: v.to(device) for k, v in self.task_logits.items()}

        # Return new output object with moved tensors
        return HmoeOutput(
            task_logits=moved_logits,
            routing_loss=self.routing_loss.to(device)
        )