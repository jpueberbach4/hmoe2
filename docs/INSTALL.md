# HMoE2 System Setup & Execution Guide

The Hierarchical Mixture of Experts (HMoE2) engine requires a highly specific environment to function correctly. Because the architecture relies on dynamic routing across multiple GPUs and geometric path signature calculations, you must configure a custom CUDA environment and a local data API before running the training loops.

---

## 1. Hardware & System Prerequisites

* **Operating System:** Linux (Ubuntu 20.04 or 22.04 strongly recommended).
* **Hardware:** An NVIDIA GPU with CUDA support is strictly required. Verify your GPU is accessible by running `nvidia-smi` in your terminal.

---

## 2. Environment Setup

You must build the environment manually to ensure PyTorch and the C++ geometric path signature dependencies compile correctly against your specific GPU architecture.

### Step 2.1: System Dependencies
Install the required system compilers and networking tools:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libomp-dev wget gpg python3-venv python3-pip python3-dev libnccl-dev
```

*Note: Certain C++ dependencies require **CMake 3.18 or higher**. If your Ubuntu distribution defaults to an older version, you must upgrade it via the Kitware repository before proceeding.*

### Step 2.2: Python Virtual Environment
Create and activate an isolated Python environment to prevent dependency conflicts:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel ninja
```

### Step 2.3: Core Machine Learning Libraries
Install PyTorch compiled specifically for CUDA 12.1, followed by the core dependencies for sequence routing and geometric transformations:
```bash
# 1. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. Install specific mathematical and data dependencies
pip install signatory requests pandas numpy plotly PyYAML PyArrow
```

---

## 3. Data Infrastructure (The Local API)

The HMoE2 engine does **not** read CSV files directly. To manage memory and allow for dynamic feature selection across thousands of epochs, the engine queries a local timeseries API.

You must expose a local REST API (defaulting to `http://localhost:8000`) that responds to the following endpoint structure:
```text
/ohlcv/1.1/select/{symbol},{timeframe}[{field1}:{field2}]/after/{timestamp_ms}/until/{timestamp_ms}/output/JSON
```

**Required API Response Format:**
The API must return a JSON object containing a `result` dictionary, where keys are the feature names and values are chronological arrays (lists) of floats. 

*Example Response:*
```json
{
  "result": {
    "open": [1.2501, 1.2505, 1.2498],
    "close": [1.2504, 1.2502, 1.2510],
    "forward-panel_1W_rsi_14_3": [55.2, 56.1, 54.8],
    "target_macro_regime": [1.0, 1.0, 0.0]
  }
}
```
*Note: Your manual ground-truth labels (e.g., `target_macro_regime`) must be ingested into your database and served alongside the standard OHLCV and indicator data.*

---

## 4. Execution Pipeline

Once your environment is compiled and your data API is live, you can execute the engine.

### Step 4.1: Determinism

Store as `determinism.py`

```python
import torch
import numpy as np
import random
import os

def set_determinism(seed=42):
    """Locks all random number generators for reproducible training."""
    # 1. Python & Standard Library
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch Core
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        
        # 4. The CUDA Execution Engine (The bottleneck)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call this immediately at the start of your script
set_determinism(1)
```

### Step 4.2: Training the Engine
To begin the training loop, pass your topology definition to the training script. The engine will automatically query your API, sanitize the data, and apply differential learning rates to the routing motifs.

```python
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import torch
import determinism

from hmoe2.loss import HmoeLossEngine
from hmoe2.nodes import HmoeNode
from hmoe2.sanitize import HmoeSanitizer
from hmoe2.schema import HmoeFeature
from hmoe2.tensor import HmoeTensor
from hmoe2.trainer import HmoeTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True 
)
logger = logging.getLogger("Trainer")


def fetch_raw_api_data(
    symbol: str, 
    timeframe: str, 
    fields: List[str], 
    start_date: str, 
    end_date: str, 
    base_url: str = "http://localhost:8000"
) -> Dict[str, list]:
    """Fetches raw OHLCV and indicator data from the local timeseries API.

    Args:
        symbol (str): The ticker symbol to fetch (e.g., 'GBP-USD').
        timeframe (str): The timeframe resolution (e.g., '4h', '1d').
        fields (List[str]): List of specific feature fields to request.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        base_url (str, optional): Base URL of the data API. Defaults to localhost.

    Returns:
        Dict[str, list]: A dictionary mapping feature names to their data arrays.

    Raises:
        ValueError: If the API returns an empty dataset.
        requests.HTTPError: If the API request fails.
    """
    after_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    fields_str = f"[{':'.join(fields)}]"
    endpoint = f"/ohlcv/1.1/select/{symbol},{timeframe}{fields_str}/after/{after_ms}/until/{until_ms}/output/JSON"
    url = f"{base_url.rstrip('/')}{endpoint}"
    
    logger.info(f"Fetching {symbol} ({start_date} to {end_date}) with {len(fields)} features.")
    
    response = requests.get(url, params={"limit": 1000000, "subformat": 3, "order": "asc"}, timeout=10)
    response.raise_for_status()
    
    raw_result = response.json().get('result', {})
    clean_dict = {}
    
    for key, values in raw_result.items():
        clean_name = key.split('__')[0] 
        if clean_name in fields:
            clean_dict[clean_name] = values
            
    if not clean_dict:
        raise ValueError(f"API returned empty data for {symbol} between {start_date} and {end_date}.")
        
    return clean_dict


def extract_required_features(root_router: HmoeNode, task_list: list) -> List[HmoeFeature]:
    """Scans the router topology to build a unique set of required data features.

    This ensures the pipeline only fetches and sanitizes data that is explicitly 
    required by the active experts and loss targets, optimizing memory usage.

    Args:
        root_router (HmoeNode): The root node of the initialized HMoE tree.
        task_list (list): The list of active tasks containing target labels.

    Returns:
        List[HmoeFeature]: A deduplicated list of feature schemas.
    """
    features: List[HmoeFeature] = []
    seen_names = set()
    
    for feature in root_router.subtree_features:
        if feature.name not in seen_names:
            features.append(feature)
            seen_names.add(feature.name)
            
    for task in task_list:
        if hasattr(task, 'label_target') and task.label_target is not None:
            if task.label_target.name not in seen_names:
                features.append(task.label_target)
                seen_names.add(task.label_target.name)
                
    return features


def configure_optimizer(model: HmoeNode, base_lr: float = 1e-3) -> torch.optim.Optimizer:
    """Configures the AdamW optimizer with differential learning rates.

    Applies a significantly higher learning rate multiplier to the learnable motif 
    tensors. This is required because geometric stencils need to morph rapidly 
    to find structural matches, while standard routing and projection layers 
    require slower, stable gradient updates.

    Args:
        model (HmoeNode): The root HMoE model.
        base_lr (float, optional): The base learning rate. Defaults to 1e-3.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    motif_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if 'motifs' in name:
            motif_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': base_lr},
        {'params': motif_params, 'lr': base_lr * 50.0} 
    ], weight_decay=1e-5)
    
    return optimizer


def prepare_datasets(
    feature_objects: List[HmoeFeature], 
    device: torch.device
) -> Tuple[List[HmoeTensor], List[HmoeTensor]]:
    """Fetches and sanitizes the training and validation datasets.

    Args:
        feature_objects (List[HmoeFeature]): The strict schemas required for sanitization.
        device (torch.device): The target compute device for the resulting tensors.

    Returns:
        Tuple[List[HmoeTensor], List[HmoeTensor]]: The training and validation dataloaders.
    """
    feature_names = [f.name for f in feature_objects]
    
    raw_train_data = fetch_raw_api_data(
        symbol="GBP-USD", timeframe="4h", fields=feature_names, 
        start_date="2014-01-01", end_date="2023-12-31"
    )
    raw_val_data = fetch_raw_api_data(
        symbol="GBP-USD", timeframe="4h", fields=feature_names, 
        start_date="2024-01-01", end_date="2025-12-31"
    )

    logger.info("Sanitizing global data pipelines.")
    dirty_train = HmoeTensor.from_dict(raw_train_data)
    dirty_val = HmoeTensor.from_dict(raw_val_data)
    
    clean_train = HmoeSanitizer.sanitize(
        dirty_train, 
        allowed_features=feature_objects, 
        drop_nan_columns=True, 
        verbose=True
    ).to(device)
    
    clean_val = HmoeSanitizer.sanitize(
        dirty_val, 
        allowed_features=feature_objects, 
        drop_nan_columns=True, 
        verbose=False
    ).to(device)

    # Wrap in lists to simulate a standard dataloader structure for the HmoeTrainer
    return [clean_train], [clean_val]


def main():
    """Main orchestrator for training the HMoE2 network."""
    parser = argparse.ArgumentParser(description="Train HMoE2 Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML topology.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device selected: {device}")

    logger.info(f"Initializing architecture from {args.config}")
    root_router = HmoeNode.from_yaml(args.config)

    global_tasks = {}
    root_router._gather_tasks(global_tasks)
    task_list = list(global_tasks.values())

    root_router = root_router.to(device)
    loss_engine = HmoeLossEngine(tasks=task_list).to(device)
    optimizer = configure_optimizer(root_router)

    required_feature_objects = extract_required_features(root_router, task_list)
    train_dataloader, val_dataloader = prepare_datasets(required_feature_objects, device)

    logger.info("Initializing HMoE trainer loop.")
    trainer = HmoeTrainer(
        model=root_router,
        loss_engine=loss_engine,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="experiment_01_gbpusd"
    )

    trainer.fit(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=100000,
        patience=20000
    )


if __name__ == "__main__":
    main()
```
* **Outputs:** The trainer will create an `experiment_01_gbpusd/` directory containing `best_checkpoint.pt`, `latest_checkpoint.pt`, and your `metrics_history.json`.

### Step 4.3: Visualizing & Backtesting
To evaluate the model's structural understanding and out-of-sample performance, run a visualization script against your best checkpoint. You can dump the code into an AI and ask it to visualize the data in a certain way.