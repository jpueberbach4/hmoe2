import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import torch

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
        epochs=10000,
        patience=500
    )


if __name__ == "__main__":
    main()