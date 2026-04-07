import os
import argparse
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import deque

import determinism
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hmoe2.nodes import HmoeNode, HmoeRouter
from hmoe2.tensor import HmoeTensor, HmoeInput
from hmoe2.sanitize import HmoeSanitizer
from hmoe2.schema import HmoeCheatFeature
from trading.regime import RegimeDetector, RegimeStateDTO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True 
)
logger = logging.getLogger("Macro_Visualizer")


# ==============================================================================
# 1. TELEMETRY & REGIME DTOs
# ==============================================================================

@dataclass
class EngineContext:
    device: torch.device
    args: argparse.Namespace
    root_router: Any
    clean_master: HmoeTensor
    master_np: np.ndarray
    ohlcv_dict: dict
    feature_to_idx: dict
    input_feature_names: list
    sequence_length: int

@dataclass
class TelemetryData:
    sig_activations: Optional[np.ndarray] = None
    sig_feature_names: List[str] = None
    regime_history: Optional[List[RegimeStateDTO]] = None
    label_name: Optional[str] = None


# ==============================================================================
# 3. DATA FETCHING & INITIALIZATION
# ==============================================================================

def fetch_raw_api_data(symbol: str, timeframe: str, fields: List[str], start_date: str, end_date: str, base_url: str = "http://localhost:8000") -> Dict[str, list]:
    after_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    plot_fields = {'open', 'high', 'low', 'close'}
    all_fields = list(set(fields).union(plot_fields))
    
    fields_str = f"[{':'.join(all_fields)}]"
    endpoint = f"/ohlcv/1.1/select/{symbol},{timeframe}{fields_str}/after/{after_ms}/until/{until_ms}/output/JSON"
    url = f"{base_url.rstrip('/')}{endpoint}"

    logger.info(f"Fetching: {symbol} | {start_date} to {end_date} | {len(all_fields)} fields")
    response = requests.get(url, params={"limit": 1000000, "subformat": 3, "order": "asc"}, timeout=10)
    response.raise_for_status()
    
    raw_result = response.json().get('result', {})
    clean_dict = {}
    for key, values in raw_result.items():
        clean_name = key.split('__')[0] 
        if clean_name in all_fields:
            clean_dict[clean_name] = values
    return clean_dict


def find_target_backend(node: Any, target_types: List[str]) -> Optional[Any]:
    if getattr(node, 'core', None) and getattr(node, 'backend_type', '').upper() in target_types:
        return node
    if hasattr(node, 'branches'):
        for child in node.branches:
            result = find_target_backend(child, target_types)
            if result:
                return result
    return None


def initialize_environment(args: argparse.Namespace) -> EngineContext:
    torch.serialization.add_safe_globals([HmoeNode])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at: {args.checkpoint}")
        sys.exit(1)

    logger.info("Initializing Graph from YAML...")
    root_router = HmoeNode.from_yaml(args.config)
    
    required_feature_objects = []
    seen_names = set()
    
    for feature in root_router.subtree_features:
        if feature.name not in seen_names:
            required_feature_objects.append(feature)
            seen_names.add(feature.name)

    # Automatically fetch the label targets assigned to the heads so we can plot them
    for name, module in root_router.named_modules():
        if hasattr(module, 'task_config') and module.task_config.label_target:
            label_feat = module.task_config.label_target
            if label_feat.name not in seen_names:
                required_feature_objects.append(label_feat)
                seen_names.add(label_feat.name)

    required_feature_names = [f.name for f in required_feature_objects]
    raw_data = fetch_raw_api_data(args.symbol, "4h", required_feature_names, args.start, args.end)
    
    ohlcv_dict = {
        'open': np.array(raw_data['open'], dtype=float),
        'high': np.array(raw_data['high'], dtype=float),
        'low': np.array(raw_data['low'], dtype=float),
        'close': np.array(raw_data['close'], dtype=float),
    }

    dirty_tensor = HmoeTensor.from_dict(raw_data)
    clean_master = HmoeSanitizer.sanitize(raw_tensor=dirty_tensor, allowed_features=required_feature_objects, drop_nan_columns=True, verbose=True).to(device)
    sequence_length = clean_master.tensor.size(1)

    # ==========================================================================
    # ABSOLUTE SILENCER LOGIC (Data Leakage Prevention)
    # ==========================================================================
    
    # 1. Capture the raw, unblinded Numpy array for the UI so charts don't flatline
    master_np_unblinded = clean_master.tensor[0].cpu().numpy()

    # 2. Gather ALL features that must be hidden from the model (Cheats + Labels)
    features_to_silence = {f.name for f in root_router.subtree_features if isinstance(f, HmoeCheatFeature)}
    for name, module in root_router.named_modules():
        if hasattr(module, 'task_config') and module.task_config.label_target:
            features_to_silence.add(module.task_config.label_target.name)

    # 3. Explicitly zero out the prohibited columns in the tensor math
    blinded_math = clean_master.tensor.clone()
    for col_idx, feature in enumerate(clean_master.indices):
        if feature.name in features_to_silence:
            blinded_math[:, :, col_idx] = 0.0

    # 4. Re-wrap the blinded tensor for inference
    clean_master = HmoeTensor(tensor=blinded_math, indices=clean_master.indices)
    feature_to_idx = {f.name: idx for idx, f in enumerate(clean_master.indices)}
    
    input_feature_names = [f.name for f in root_router.subtree_features if not isinstance(f, HmoeCheatFeature)]
    input_feature_names = list(dict.fromkeys(input_feature_names)) 

    root_router.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    root_router.load_state_dict(checkpoint['model_state_dict'])
    root_router.eval()
    logger.info(f"Successfully loaded checkpoint: {args.checkpoint}")

    return EngineContext(
        device=device,
        args=args,
        root_router=root_router,
        clean_master=clean_master,
        master_np=master_np_unblinded, # Passes the pure unblinded data to the UI
        ohlcv_dict=ohlcv_dict,
        feature_to_idx=feature_to_idx,
        input_feature_names=input_feature_names,
        sequence_length=sequence_length
    )


# ==============================================================================
# 4. TELEMETRY EXTRACTION
# ==============================================================================

def execute_telemetry_pass(ctx: EngineContext) -> TelemetryData:
    target_task = "task_regime"
    target_head = None
    
    for name, module in ctx.root_router.named_modules():
        if name.endswith(target_task):
            target_head = module
            logger.info(f"Found target head for '{target_task}' at: {name}")
            break
            
    if target_head is None:
        logger.error(f"Could not find a head module ending with '{target_task}'.")
        target_head = ctx.root_router 

    # Extract the actual label assigned to this task head
    label_name = None
    if hasattr(target_head, 'task_config') and target_head.task_config.label_target:
        label_name = target_head.task_config.label_target.name
        
    regime_detector = RegimeDetector(
        target_module=target_head,
        lookback_window=1000,
        sma_window=5,          
        high_threshold=0.50,  
        low_threshold=0.2,    
        entry_debounce=2,      
        exit_debounce=3        
    )

    logger.info("Executing baseline inference pass for Macro regime detection...")
    with torch.no_grad():
        payload = HmoeInput(tensor=ctx.clean_master)
        _ = ctx.root_router(payload) 
    
    regime_history = list(regime_detector.state_history)
    
    densities = np.array([state.smoothed_density for state in regime_history])
    sig_activations = densities.reshape(-1, 1)
    
    regime_detector.remove_listener()

    return TelemetryData(
        sig_activations=sig_activations,
        sig_feature_names=[f"{target_task}_confidence"],
        regime_history=regime_history,
        label_name=label_name
    )


# ==============================================================================
# 5. UI RENDERING (PURE MACRO VIEW)
# ==============================================================================

def render_macro_dashboard(ctx: EngineContext, tel: TelemetryData) -> None:
    has_sig = tel.sig_activations is not None and tel.regime_history is not None
    has_label = tel.label_name is not None and tel.label_name in ctx.feature_to_idx

    num_indicators = len(ctx.input_feature_names)
    
    # Calculate rows based on available data
    total_rows = 1 + num_indicators + (2 if has_sig else 0) + (1 if has_label else 0)
    
    row_heights = [4.0] + [1.0] * num_indicators
    if has_sig:
        row_heights.extend([0.8, 1.5])
    if has_label:
        row_heights.append(1.0)
    
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=row_heights)
    x_indices = np.arange(ctx.sequence_length)

    # 1. Plot Price
    fig.add_trace(go.Candlestick(x=x_indices, open=ctx.ohlcv_dict["open"], high=ctx.ohlcv_dict["high"], low=ctx.ohlcv_dict["low"], close=ctx.ohlcv_dict["close"], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    # 2. Plot Input Indicators
    current_row_idx = 1
    for f_name in ctx.input_feature_names:
        current_row_idx += 1
        if f_name in ctx.feature_to_idx:
            f_data = ctx.master_np[:, ctx.feature_to_idx[f_name]]
            fig.add_trace(go.Scatter(x=x_indices, y=f_data, mode='lines', name=f_name, line=dict(width=1), hovertemplate="%{y:.4f}<extra></extra>"), row=current_row_idx, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="white", row=current_row_idx, col=1, opacity=0.2)
            fig.add_annotation(x=0.01, y=0.95, xref=f"x{current_row_idx} domain", yref=f"y{current_row_idx} domain", text=f"<b>{f_name}</b>", showarrow=False, font=dict(color="white", size=11), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    # 3. Plot Signature X-Ray and Regime Overlay
    if has_sig:
        barcode_row = current_row_idx + 1
        conf_row = current_row_idx + 2
        current_row_idx += 2
        
        sig_z_vis = tel.sig_activations.T

        # Map Continuous Uptrend / Downtrend Blocks
        seq_len = len(tel.regime_history)
        regime_blocks = []
        
        current_state = tel.regime_history[0].is_hunting_zone
        last_idx = 0
        
        for t in range(1, seq_len):
            state = tel.regime_history[t].is_hunting_zone
            if state != current_state:
                regime_blocks.append((last_idx, t, current_state))
                current_state = state
                last_idx = t
                
        # Append the final block
        regime_blocks.append((last_idx, seq_len, current_state))

        # Draw continuous alternating regime blocks across the entire chart
        for start_idx, end_idx, is_hunting in regime_blocks:
            if is_hunting:
                # Uptrend -> Green overlay
                fill_color = "rgba(0, 255, 0, 0.08)"
                line_color = "rgba(0, 255, 0, 0.5)"
            else:
                # Downtrend -> Red overlay
                fill_color = "rgba(255, 0, 0, 0.08)"
                line_color = "rgba(255, 0, 0, 0.5)"
                
            fig.add_vrect(
                x0=start_idx, x1=end_idx, 
                fillcolor=fill_color, layer="below", line_width=1, line_color=line_color,
                row="all", col=1
            )
        
        # DEDICATED PANEL 1: Render Pure Barcode Heatmap
        replicated_colorscale = [
            [0.0, "rgb(0, 0, 0)"], [0.3, "rgb(0, 0, 0)"], [0.5, "rgb(50, 20, 0)"],       
            [0.75, "rgb(200, 120, 0)"], [0.9, "rgb(255, 215, 0)"], [1.0, "rgb(255, 255, 200)"]    
        ]
        
        fig.add_trace(go.Heatmap(z=sig_z_vis, x=x_indices, y=[0], colorscale=replicated_colorscale, showscale=False, hoverinfo='skip', name="Confidence Barcode"), row=barcode_row, col=1)
        fig.update_yaxes(showticklabels=False, row=barcode_row, col=1)
        
        # DEDICATED PANEL 2: Render Pure Confidence Line
        smoothed_counts = [state.smoothed_density for state in tel.regime_history]
        fig.add_trace(go.Scatter(x=x_indices, y=smoothed_counts, mode='lines', line=dict(color='cyan', width=2), name="Pure Confidence"), row=conf_row, col=1)
        
        fig.add_hline(y=0.50, line_dash="dash", line_color="rgba(0, 255, 0, 0.8)", row=conf_row, col=1)
        fig.add_hline(y=0.30, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)", row=conf_row, col=1)
        
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{conf_row} domain", yref=f"y{conf_row} domain", text=f"<b>Pure Confidence ({', '.join(tel.sig_feature_names)})</b>", showarrow=False, font=dict(color="cyan", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    # 4. Plot Ground Truth Label (Bottom Panel)
    if has_label:
        label_row = current_row_idx + 1
        label_data = ctx.master_np[:, ctx.feature_to_idx[tel.label_name]]
        fig.add_trace(go.Scatter(x=x_indices, y=label_data, mode='lines', name="Ground Truth", line=dict(color='magenta', width=1), fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'), row=label_row, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=label_row, col=1)
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{label_row} domain", yref=f"y{label_row} domain", text=f"<b>Label Target: {tel.label_name}</b>", showarrow=False, font=dict(color="magenta", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    calc_height = 400 + (150 * num_indicators) + (350 if has_sig else 0) + (150 if has_label else 0)
    title_str = f"HMoE2 Pure Macro Regime Viewer (Uptrend Hunter) | {ctx.args.symbol}"
    
    fig.update_layout(title=title_str, template="plotly_dark", hovermode="x unified", height=calc_height, xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=60, b=40))
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="HMoE2 Macro Regime Visualizer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML topology.")
    parser.add_argument("--checkpoint", type=str, default="experiment_01_gbpusd/best_checkpoint.pt", help="Path to the .pt checkpoint file.")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-06-01", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="XAU-USD", help="Ticker Symbol")
    args = parser.parse_args()

    ctx = initialize_environment(args)
    tel = execute_telemetry_pass(ctx)
    render_macro_dashboard(ctx, tel)


if __name__ == "__main__":
    main()