import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Any

from hmoe2.schema import HmoeTask, HmoeNodeType
from hmoe2.tensor import HmoeTensor, HmoeInput, HmoeOutput
from hmoe2.heads import HmoeHead
from hmoe2.nodes import HmoeNode

from hmoe2.backends import (
    LinearBackend, TcnBackend, GruBackend,
    CausalTransformerBackend, GatedResidualBackend, LstmBackend,
    RnnBackend, SignatureBackend, MotifsBackend
)


@dataclass(eq=False)
class HmoeExpert(HmoeNode):
    """Leaf node representing an expert in a hierarchical Mixture-of-Experts model.

    This class dynamically instantiates a neural backend based on configuration
    and routes feature subsets through the selected backend. It also manages
    task-specific heads for producing outputs.

    Attributes:
        features (list): List of feature descriptors used by this expert.
        allowed_tasks (list): List of tasks this expert is responsible for.
        backend_type (str): String identifier of the backend type.
        input_dim (int): Number of input features.
        hidden_dim (int): Hidden representation size.
        core (nn.Module): Instantiated backend model.
        task_heads (nn.ModuleDict): Mapping of task names to task-specific heads.
    """

    def __init__(self, name: str, tasks: list, features: list, backend: str = "LINEAR", hidden_dim: int = 32, config: Dict = {}):
        """Initializes the HmoeExpert.

        Args:
            name (str): Unique name of the expert node.
            tasks (list): List of HmoeTask objects assigned to this expert.
            features (list): List of feature descriptors used as input.
            backend (str): Backend type identifier (e.g., LINEAR, TCN, GRU).
            hidden_dim (int): Dimensionality of hidden representations.
        """
        nn.Module.__init__(self)
        super().__init__(name=name, type=HmoeNodeType.EXPERT, features=features)

        # Store feature configuration and task assignments
        self.features = features
        self.allowed_tasks = tasks

        # Normalize backend type string for consistent matching
        self.backend_type = backend.upper()

        # Determine input dimensionality from feature list
        self.input_dim = len(self.features)

        # Store hidden dimension size for backend and heads
        self.hidden_dim = hidden_dim

        # Rest of config
        self.config = config

        # Dynamically instantiate the appropriate backend based on configuration
        if self.backend_type == "LINEAR":
            self.core = LinearBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type == "TCN":
            self.core = TcnBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type == "GRU":
            self.core = GruBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type == "LSTM":
            self.core = LstmBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type in ["GATED_RESIDUAL", "GR"]:
            self.core = GatedResidualBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type in ["CAUSAL_TRANSFORMER", "TRANSFORMER", "CT"]:
            self.core = CausalTransformerBackend(self.input_dim, self.hidden_dim, config=config)
        elif self.backend_type in ["VANILLA_RNN", "RNN"]:
            self.core = RnnBackend(self.input_dim, self.hidden_dim, config=config)
        # Experimental (TODO: add parameterization)
        elif self.backend_type in ["SIGNATURE", "SIGNATORY", "ROUGH_PATH", "RP"]:
            self.core = SignatureBackend(
                input_dim=self.input_dim, 
                hidden_dim=self.hidden_dim,
                config=config
            )
        elif self.backend_type in ["MATRIX_PROFILE", "MOTIF", "MP"]:
            self.core = MotifsBackend(
                self.input_dim, 
                self.hidden_dim,
                config=config
            )
        else:
            # Raise an error if an unknown backend type is provided
            raise ValueError(f"Unknown backend type: {self.backend_type} in expert {name}")

        # Initialize container for task-specific output heads
        self.task_heads = nn.ModuleDict()

    def _gather_tasks(self, task_dict: Dict[str, HmoeTask]) -> None:
        """Collects and registers tasks into a shared task dictionary.

        Args:
            task_dict (Dict[str, HmoeTask]): Dictionary mapping task names to task objects.
        """
        # Iterate through allowed tasks and add them to the dictionary if missing
        for task_obj in self.allowed_tasks:
            if task_obj.name not in task_dict:
                task_dict[task_obj.name] = task_obj

    def _serialize_node(self) -> dict:
        """Serializes the expert configuration into a dictionary.

        Returns:
            dict: Serialized representation of the expert node.
        """
        # Construct a dictionary capturing all relevant configuration fields
        config_dict = {
            "name": self.name,
            "type": "EXPERT",
            "backend": self.backend_type,
            "hidden_dim": self.hidden_dim,
            "allowed_tasks": [t.name for t in self.allowed_tasks],
            "features": [f.serialize() for f in self.features]
        }
        if self.dilations is not None and self.backend_type == "TCN":
            config_dict["dilations"] = self.dilations

        return config_dict

    def link_tasks(self, global_tasks: List[HmoeTask]) -> None:
        """Initializes task-specific heads based on assigned tasks.

        Args:
            global_tasks (List[HmoeTask]): List of globally defined tasks.
        """
        # Create a prediction head for each allowed task
        for task_obj in self.allowed_tasks:
            self.task_heads[task_obj.name] = HmoeHead(
                input_dim=self.hidden_dim,
                task_config=task_obj
            )

    def forward(self, payload: HmoeInput) -> HmoeOutput:
        """Executes forward pass through the expert.

        Args:
            payload (HmoeInput): Input payload containing feature data.

        Returns:
            HmoeOutput: Output containing task-specific logits.
        """
        # Extract only the subset of features relevant to this expert
        narrowed_payload = payload.get_subset(self.features)

        # Convert structured payload into tensor representation
        raw_data = narrowed_payload.to_tensor()

        # Pass data through the selected backend model
        hidden_state = self.core(raw_data)

        # Wrap hidden representation into HmoeTensor abstraction
        hidden_tensor = HmoeTensor(tensor=hidden_state, indices=[])

        # Compute outputs for each task head
        task_logits = {}
        for task_name, head in self.task_heads.items():
            task_logits[task_name] = head(hidden_tensor)

        # Return structured output containing all task predictions
        return HmoeOutput(task_logits=task_logits)