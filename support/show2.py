"""
THIS IS A DEMO

INCLUDES WALKFORWARD TEST USING MACRO ENTRIES ONLY

IT ALSO SHOWS HOW TO USE MULTIPLE HEADS AND HOW TO USE THEM IN A SINGLE INTERFACE

THIS DEMO USES CONFIG2.YAML

WHEN PROPERLY CONFIGURED THIS SHOULD GIVE WR'S OF ABOUT 60-70 PERCENT (FOREX)
WHEN YOU DONT GET THESE RATES. CHECK YOUR DATA PIPELINE. YOU NEED 4H, 8H, 1D AND 1W DATA.

IF YOU HAVE THE BP.MARKETS.INGEST PROJECT. COPY OVER THE BP DIRECTORY TO YOUR CONFIG.USER.

BP MARKETS WILL RETURN BUT IS UNDERGOING A MAJOR OVERHAUL
"""
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
    prob_tops: Optional[np.ndarray] = None      # <--- NEW
    prob_bottoms: Optional[np.ndarray] = None   # <--- NEW


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
    
    master_np_unblinded = clean_master.tensor[0].cpu().numpy()

    features_to_silence = {f.name for f in root_router.subtree_features if isinstance(f, HmoeCheatFeature)}
    for name, module in root_router.named_modules():
        if hasattr(module, 'task_config') and module.task_config.label_target:
            features_to_silence.add(module.task_config.label_target.name)

    blinded_math = clean_master.tensor.clone()
    for col_idx, feature in enumerate(clean_master.indices):
        if feature.name in features_to_silence:
            blinded_math[:, :, col_idx] = 0.0

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
        master_np=master_np_unblinded,
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

    label_name = None
    if hasattr(target_head, 'task_config') and target_head.task_config.label_target:
        label_name = target_head.task_config.label_target.name
        
    regime_detector = RegimeDetector(
        target_module=target_head,
        lookback_window=1000,
        sma_window=5,          
        high_threshold=0.50,  
        low_threshold=0.1,    
        entry_debounce=2,      
        exit_debounce=3        
    )

    logger.info("Executing baseline inference pass for Macro regime & Micro tasks...")
    with torch.no_grad():
        payload = HmoeInput(tensor=ctx.clean_master)
        # Capture the full output payload to grab multihead logits
        preds = ctx.root_router(payload) 
    
    # 1. Macro Telemetry
    regime_history = list(regime_detector.state_history)
    densities = np.array([state.smoothed_density for state in regime_history])
    sig_activations = densities.reshape(-1, 1)
    regime_detector.remove_listener()

    # 2. Micro Tasks Telemetry
    prob_tops = None
    prob_bottoms = None
    
    if 'task_tops' in preds.task_logits:
        logits = preds.task_logits['task_tops'].to_tensor()
        prob_tops = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        
    if 'task_bottoms' in preds.task_logits:
        logits = preds.task_logits['task_bottoms'].to_tensor()
        prob_bottoms = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

    # ==========================================================================
    # 3. MACRO-TO-MICRO SIGNAL MASKING
    # ==========================================================================
    # Extract boolean array: True = Uptrend (Green Block), False = Downtrend (Black Block)
    is_bull_regime = np.array([state.is_hunting_zone for state in regime_history], dtype=bool)
    
    # If in an uptrend, completely kill the 'Tops' confidence (flatten to 0.0)
    if prob_tops is not None:
        prob_tops[is_bull_regime] = 0.0
        
    # If in a downtrend, completely kill the 'Bottoms' confidence (flatten to 0.0)
    if prob_bottoms is not None:
        prob_bottoms[~is_bull_regime] = 0.0

    return TelemetryData(
        sig_activations=sig_activations,
        sig_feature_names=[f"{target_task}_confidence"],
        regime_history=regime_history,
        label_name=label_name,
        prob_tops=prob_tops,
        prob_bottoms=prob_bottoms
    )


# ==============================================================================
# 5. UI RENDERING 
# ==============================================================================

def render_macro_dashboard(ctx: EngineContext, tel: TelemetryData) -> None:
    has_sig = tel.sig_activations is not None and tel.regime_history is not None
    has_label = tel.label_name is not None and tel.label_name in ctx.feature_to_idx
    has_micro = tel.prob_tops is not None or tel.prob_bottoms is not None

    num_indicators = len(ctx.input_feature_names)
    
    # Calculate rows based on available data
    total_rows = 1 + num_indicators + (2 if has_sig else 0) + (1 if has_micro else 0) + (1 if has_label else 0)
    
    row_heights = [4.0] + [1.0] * num_indicators
    if has_sig:
        row_heights.extend([0.8, 1.5])
    if has_micro:
        row_heights.append(1.5)
    if has_label:
        row_heights.append(1.0)
    
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=row_heights)
    x_indices = np.arange(ctx.sequence_length)

    # 1. Plot Price
    fig.add_trace(go.Candlestick(x=x_indices, open=ctx.ohlcv_dict["open"], high=ctx.ohlcv_dict["high"], low=ctx.ohlcv_dict["low"], close=ctx.ohlcv_dict["close"], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    # NEW: Overlay Micro Top/Bottom Signals on Price Chart
    if has_micro:
        offset_dist = np.nanmean(ctx.ohlcv_dict["close"]) * 0.002
        if tel.prob_tops is not None:
            top_idx = np.where(tel.prob_tops >= 0.80)[0]
            fig.add_trace(go.Scatter(x=top_idx, y=ctx.ohlcv_dict["high"][top_idx] + offset_dist, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='black')), name='Predicted Top'), row=1, col=1)
            
        if tel.prob_bottoms is not None:
            bot_idx = np.where(tel.prob_bottoms >= 0.80)[0]
            fig.add_trace(go.Scatter(x=bot_idx, y=ctx.ohlcv_dict["low"][bot_idx] - offset_dist, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(width=1, color='black')), name='Predicted Bottom'), row=1, col=1)

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
                
        regime_blocks.append((last_idx, seq_len, current_state))

        for start_idx, end_idx, is_hunting in regime_blocks:
            if is_hunting:
                fill_color = "rgba(0, 255, 0, 0.08)"
                
                fig.add_vrect(
                    x0=start_idx, x1=end_idx, 
                    fillcolor=fill_color, layer="below", line_width=0,
                    row="all", col=1
                )
                
                fig.add_vline(
                    x=start_idx, line_width=2, line_dash="solid", line_color="rgba(0, 255, 0, 0.8)",
                    row="all", col=1
                )
                
                if end_idx < seq_len:
                    fig.add_vline(
                        x=end_idx, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)",
                        row="all", col=1
                    )
        
        replicated_colorscale = [
            [0.0, "rgb(0, 0, 0)"], [0.3, "rgb(0, 0, 0)"], [0.5, "rgb(50, 20, 0)"],       
            [0.75, "rgb(200, 120, 0)"], [0.9, "rgb(255, 215, 0)"], [1.0, "rgb(255, 255, 200)"]    
        ]
        
        fig.add_trace(go.Heatmap(z=sig_z_vis, x=x_indices, y=[0], colorscale=replicated_colorscale, showscale=False, hoverinfo='skip', name="Confidence Barcode"), row=barcode_row, col=1)
        fig.update_yaxes(showticklabels=False, row=barcode_row, col=1)
        
        smoothed_counts = [state.smoothed_density for state in tel.regime_history]
        fig.add_trace(go.Scatter(x=x_indices, y=smoothed_counts, mode='lines', line=dict(color='cyan', width=2), name="Pure Confidence"), row=conf_row, col=1)
        
        fig.add_hline(y=0.50, line_dash="dash", line_color="rgba(0, 255, 0, 0.8)", row=conf_row, col=1)
        fig.add_hline(y=0.30, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)", row=conf_row, col=1)
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{conf_row} domain", yref=f"y{conf_row} domain", text=f"<b>Macro Regime Confidence ({', '.join(tel.sig_feature_names)})</b>", showarrow=False, font=dict(color="cyan", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    # 4. NEW: Plot Micro Task Probabilities
    if has_micro:
        micro_row = current_row_idx + 1
        current_row_idx += 1
        
        if tel.prob_tops is not None:
            fig.add_trace(go.Scatter(x=x_indices, y=tel.prob_tops, mode='lines', line=dict(color='red', width=1.5), name="Prob Top"), row=micro_row, col=1)
        if tel.prob_bottoms is not None:
            fig.add_trace(go.Scatter(x=x_indices, y=tel.prob_bottoms, mode='lines', line=dict(color='lime', width=1.5), name="Prob Bottom"), row=micro_row, col=1)
            
        fig.add_hline(y=0.80, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", row=micro_row, col=1)
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{micro_row} domain", yref=f"y{micro_row} domain", text="<b>Micro Tasks (Tops / Bottoms) Confidence</b>", showarrow=False, font=dict(color="white", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    # 5. Plot Ground Truth Label
    if has_label:
        label_row = current_row_idx + 1
        label_data = ctx.master_np[:, ctx.feature_to_idx[tel.label_name]]
        fig.add_trace(go.Scatter(x=x_indices, y=label_data, mode='lines', name="Ground Truth", line=dict(color='magenta', width=1), fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'), row=label_row, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=label_row, col=1)
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{label_row} domain", yref=f"y{label_row} domain", text=f"<b>Label Target: {tel.label_name}</b>", showarrow=False, font=dict(color="magenta", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    calc_height = 400 + (150 * num_indicators) + (350 if has_sig else 0) + (150 if has_micro else 0) + (150 if has_label else 0)
    title_str = f"HMoE2 Universal Trading Engine | {ctx.args.symbol}"
    
    fig.update_layout(title=title_str, template="plotly_dark", hovermode="x unified", height=calc_height, xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=60, b=40))
    fig.show()


def run_walkforward_backtest(ctx: EngineContext, tel: TelemetryData):
    logger.info("Initializing Walk-Forward Backtest Engine...")
    
    highs = ctx.ohlcv_dict['high']
    lows = ctx.ohlcv_dict['low']
    closes = ctx.ohlcv_dict['close']
    seq_len = ctx.sequence_length
    
    # 1. Execute Trade Simulation
    trades = []
    active_trade = None
    trailing_pct = 0.02  # 2% Trailing Stop
    
    for t in range(1, seq_len):
        # --- MANAGE ACTIVE TRADE ---
        if active_trade is not None:
            current_high = highs[t]
            current_low = lows[t]
            
            if active_trade['type'] == 'LONG':
                # Hit Trailing Stop
                if current_low <= active_trade['sl']:
                    active_trade['pnl_r'] = (active_trade['sl'] - active_trade['entry_price']) / active_trade['base_risk']
                    trades.append(active_trade)
                    active_trade = None
                    continue
                    
                # Trail Logic: 2% behind the highest high
                potential_sl = current_high * (1.0 - trailing_pct)
                if potential_sl > active_trade['sl']:
                    active_trade['sl'] = potential_sl
                        
            elif active_trade['type'] == 'SHORT':
                # Hit Trailing Stop
                if current_high >= active_trade['sl']:
                    active_trade['pnl_r'] = (active_trade['entry_price'] - active_trade['sl']) / active_trade['base_risk']
                    trades.append(active_trade)
                    active_trade = None
                    continue
                    
                # Trail Logic: 2% above the lowest low
                potential_sl = current_low * (1.0 + trailing_pct)
                if potential_sl < active_trade['sl']:
                    active_trade['sl'] = potential_sl
        
        # --- ENTRY LOGIC (Only trigger if flat) ---
        if active_trade is None:
            state_prev = tel.regime_history[t-1].is_hunting_zone
            state_curr = tel.regime_history[t].is_hunting_zone
            
            # Green Block Start -> Enter LONG
            if state_curr and not state_prev:
                entry_price = closes[t] 
                risk_val = entry_price * trailing_pct # Initial risk is 2% of entry
                
                active_trade = {
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'base_risk': risk_val,
                    'sl': entry_price - risk_val
                }
                
            # Red Block Start (Blank space) -> Enter SHORT
            elif not state_curr and state_prev:
                entry_price = closes[t]
                risk_val = entry_price * trailing_pct # Initial risk is 2% of entry
                
                active_trade = {
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'base_risk': risk_val,
                    'sl': entry_price + risk_val
                }

    # Clean up any open trade at the end of the sequence
    if active_trade is not None:
         last_close = closes[-1]
         pnl = (last_close - active_trade['entry_price']) if active_trade['type'] == 'LONG' else (active_trade['entry_price'] - last_close)
         active_trade['pnl_r'] = pnl / active_trade['base_risk']
         trades.append(active_trade)
         
    # 2. Dump Diagnostics
    wins = [t for t in trades if t['pnl_r'] > 0]
    losses = [t for t in trades if t['pnl_r'] <= 0]
    
    total_r = sum(t['pnl_r'] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD BACKTEST RESULTS (Pure Trend Follow | 2% Trailing SL)")
    logger.info("=" * 60)
    logger.info(f"Total Trades : {len(trades)}")
    logger.info(f"Win Rate     : {win_rate:.2f}%")
    logger.info(f"Net PnL (R)  : {total_r:.2f} R")
    logger.info(f"Longs        : {len([t for t in trades if t['type'] == 'LONG'])}")
    logger.info(f"Shorts       : {len([t for t in trades if t['type'] == 'SHORT'])}")
    logger.info("=" * 60 + "\n")

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

    # Run the backtest and dump stats to the terminal
    run_walkforward_backtest(ctx, tel)

    render_macro_dashboard(ctx, tel)


if __name__ == "__main__":
    main()