import torch.nn as nn
from hmoe2.schema import HmoeTask, HmoeFeature
from hmoe2.tensor import HmoeTensor


class HmoeHead(nn.Module):
    """Task-specific prediction head for HMoE models.

    This module maps a shared latent representation into task-specific
    output logits using a simple linear projection. It operates strictly
    on HmoeTensor abstractions to maintain schema consistency.

    Attributes:
        classifier (nn.Linear): Linear layer mapping hidden features to class logits.
        task_config (HmoeTask): Configuration object defining the task.
        output_features (list): List of output feature descriptors corresponding to classes.
    """

    def __init__(self, input_dim: int, task_config: HmoeTask):
        """Initializes the HmoeHead.

        Args:
            input_dim (int): Dimensionality of the shared representation.
            task_config (HmoeTask): Task configuration containing metadata such as number of classes.
        """
        super().__init__()

        # Define linear classifier mapping hidden representation to task logits
        self.classifier = nn.Linear(input_dim, task_config.num_classes)

        # Store task configuration for reference
        self.task_config = task_config

        # Create feature descriptors for each output class
        self.output_features = [
            HmoeFeature(name=f"{task_config.name}_class_{i}")
            for i in range(task_config.num_classes)
        ]

    def forward(self, shared_representation: HmoeTensor) -> HmoeTensor:
        """Generates task-specific logits from shared representation.

        Args:
            shared_representation (HmoeTensor): Input tensor containing shared latent features.

        Returns:
            HmoeTensor: Output tensor containing logits and corresponding feature indices.
        """
        # Extract raw tensor data from the structured HmoeTensor
        raw_tensor = shared_representation.to_tensor()

        # Apply linear classifier to obtain logits for each class
        raw_logits = self.classifier(raw_tensor)

        # Wrap logits back into HmoeTensor with associated feature metadata
        return HmoeTensor(
            tensor=raw_logits,
            indices=self.output_features
        )