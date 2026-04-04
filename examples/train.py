import time
import requests
import torch
import argparse  # <-- ADDED
from typing import List, Dict
from datetime import datetime, timezone

# HMOE Framework Imports
from hmoe2.nodes import HmoeNode
from hmoe2.loss import HmoeLossEngine, HmoeLossResult
from hmoe2.tensor import HmoeTensor, HmoeInput, HmoeOutput
from hmoe2.sanitize import HmoeSanitizer
from hmoe2.trainer import HmoeTrainer
from hmoe2.schema import HmoeFeature

def fetch_raw_api_data(
    symbol: str, timeframe: str, fields: List[str], start_date: str, end_date: str, base_url: str = "http://localhost:8000"
) -> Dict[str, list]:
    after_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    fields_str = f"[{':'.join(fields)}]"
    endpoint = f"/ohlcv/1.1/select/{symbol},{timeframe}{fields_str}/after/{after_ms}/until/{until_ms}/output/JSON"
    url = f"{base_url.rstrip('/')}{endpoint}"
    print(f"[*] Fetching: {symbol} | {start_date} to {end_date} | {len(fields)} fields")
    
    response = requests.get(url, params={"limit": 1000000, "subformat": 3, "order": "asc"}, timeout=10)
    response.raise_for_status()
    
    raw_result = response.json().get('result', {})
    clean_dict = {}
    for key, values in raw_result.items():
        clean_name = key.split('__')[0] 
        if clean_name in fields:
            clean_dict[clean_name] = values
    if not clean_dict:
        raise ValueError(f"API returned empty data.")
    return clean_dict

def extract_required_features(root_router: HmoeNode, task_list: list) -> List[HmoeFeature]:
    """Extracts unique HmoeFeature objects from the router tree and tasks."""
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

def main():
    # =========================================================================
    # ARGUMENT PARSER
    # =========================================================================
    parser = argparse.ArgumentParser(description="Train HMoE2 Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    # You can easily add more flags here later (e.g., --symbol, --epochs)
    args = parser.parse_args()

    print(f"=== 1. Architectural Initialization ===")
    print(f"[*] Loading config from: {args.config}")
    
    # Load from the parsed argument instead of hardcoded string
    root_router = HmoeNode.from_yaml(args.config)

    global_tasks = {}
    root_router._gather_tasks(global_tasks)
    task_list = list(global_tasks.values())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_router = root_router.to(device)
    loss_engine = HmoeLossEngine(tasks=task_list).to(device)
    optimizer = torch.optim.Adam(root_router.parameters(), lr=1e-3, weight_decay=1e-5)

    # Extract strict objects
    required_feature_objects = extract_required_features(root_router, task_list)
    # Derive string names strictly for the API request payload
    required_feature_names = [f.name for f in required_feature_objects]
    
    print("\n=== 2. Market Data Ingestion ===")
    raw_train_data = fetch_raw_api_data(
        symbol="GBP-USD", timeframe="4h", fields=required_feature_names, start_date="2014-01-01", end_date="2023-12-31"
    )
    raw_val_data = fetch_raw_api_data(
        symbol="GBP-USD", timeframe="4h", fields=required_feature_names, start_date="2024-01-01", end_date="2025-12-31"
    )

    print("\n=== 3. Global Data Sanitization ===")
    dirty_train = HmoeTensor.from_dict(raw_train_data)
    dirty_val = HmoeTensor.from_dict(raw_val_data)
    
    print("[*] Sanitizing Training Set...")
    # Passing the List[HmoeFeature]
    clean_train = HmoeSanitizer.sanitize(
        dirty_train, 
        allowed_features=required_feature_objects, 
        drop_nan_columns=True, 
        verbose=True
    )
    
    print("\n[*] Sanitizing Validation Set...")
    clean_val = HmoeSanitizer.sanitize(
        dirty_val, 
        allowed_features=required_feature_objects, 
        drop_nan_columns=True, 
        verbose=False
    )

    train_dataloader = [clean_train]
    val_dataloader = [clean_val]

    print("\n=== 4. Hmoe Trainer Execution ===")
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
        patience=100
    )

if __name__ == "__main__":
    main()