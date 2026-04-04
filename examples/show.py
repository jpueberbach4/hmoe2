import os
import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List
from datetime import datetime, timezone
import requests

# HMOE Framework Imports
from hmoe2.nodes import HmoeNode, HmoeRouter
from hmoe2.tensor import HmoeTensor, HmoeInput
from hmoe2.sanitize import HmoeSanitizer
from hmoe2.schema import HmoeFeature, HmoeCheatFeature

# ==========================================
# DIAGNOSTIC LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True 
)
logger = logging.getLogger("Visualizer")

# ==========================================
# DATA INGESTION
# ==========================================
def fetch_raw_api_data(
    symbol: str, 
    timeframe: str, 
    fields: List[str], 
    start_date: str, 
    end_date: str, 
    base_url: str = "http://localhost:8000"
) -> Dict[str, list]:
    after_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Force OHLCV inclusion for plotting purposes
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

# =============================================================================
# MAIN INFERENCE AND VISUALIZATION ROUTINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HMoE2 Strict Inference Visualizer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML topology.")
    parser.add_argument("--checkpoint", type=str, default="experiment_01_gbpusd/best_checkpoint.pt", help="Path to the .pt checkpoint file.")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-05-01", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="GBP-USD", help="Ticker Symbol")
    parser.add_argument("--sell-threshold", type=float, default=0.5, help="Override SELL threshold.")
    parser.add_argument("--buy-threshold", type=float, default=0.5, help="Override BUY threshold.")
    parser.add_argument("--bear-penalty", type=float, default=0.0, help="Max threshold penalty during bear markets.")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at: {args.checkpoint}")
        return

    # 1. Build Architecture & Identify Requirements
    logger.info("Initializing Graph from YAML...")
    root_router = HmoeNode.from_yaml(args.config)
    
    global_tasks = {}
    root_router._gather_tasks(global_tasks)
    
    # --- STRICT OBJECT EXTRACTION ---
    required_feature_objects = []
    seen_names = set()
    
    for feature in root_router.subtree_features:
        if feature.name not in seen_names:
            required_feature_objects.append(feature)
            seen_names.add(feature.name)
            
    for task in global_tasks.values():
        if getattr(task, 'label_target', None):
            if task.label_target.name not in seen_names:
                required_feature_objects.append(task.label_target)
                seen_names.add(task.label_target.name)

    # Map back to strings purely for the API request
    required_feature_names = [f.name for f in required_feature_objects]

    # 2. Fetch Data
    raw_data = fetch_raw_api_data(args.symbol, "4h", required_feature_names, args.start, args.end)
    ohlcv_dict = {
        'open': np.array(raw_data['open'], dtype=float),
        'high': np.array(raw_data['high'], dtype=float),
        'low': np.array(raw_data['low'], dtype=float),
        'close': np.array(raw_data['close'], dtype=float),
    }

    # 3. Sanitize Pipeline
    dirty_tensor = HmoeTensor.from_dict(raw_data)
    
    # --- PASS THE VIP LIST TO THE BOUNCER ---
    clean_master = HmoeSanitizer.sanitize(
        raw_tensor=dirty_tensor, 
        allowed_features=required_feature_objects, 
        drop_nan_columns=True,
        verbose=True
    ).to(DEVICE)
    
    sequence_length = clean_master.tensor.size(1)

    # --- THE DEADBOLT: Blind the Out-Of-Sample Data ---
    cheat_feature_names = {f.name for f in root_router.subtree_features if isinstance(f, HmoeCheatFeature)}
    blinded_math = clean_master.tensor.clone()

    for col_idx, feature in enumerate(clean_master.indices):
        if feature.name in cheat_feature_names:
            logger.info(f"🔒 DEADBOLT ENGAGED: Blinding cheat feature -> {feature.name}")
            blinded_math[:, :, col_idx] = 0.0

    # Overwrite the master tensor with the blinded math
    clean_master = HmoeTensor(tensor=blinded_math, indices=clean_master.indices)

    # --- EXTRACT SANITIZED INPUT FEATURES FOR PLOTTING ---
    # Convert exactly what the neural network sees back to numpy
    master_np = clean_master.tensor[0].cpu().numpy()
    feature_to_idx = {f.name: idx for idx, f in enumerate(clean_master.indices)}
    
    # Filter out cheat features (labels) to only plot the true inputs
    input_feature_names = []
    seen_inputs = set()
    for f in root_router.subtree_features:
        if not isinstance(f, HmoeCheatFeature) and f.name not in seen_inputs:
            input_feature_names.append(f.name)
            seen_inputs.add(f.name)

    # 4. Load Weights (Now that the architecture is fully linked)
    root_router.to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    root_router.load_state_dict(checkpoint['model_state_dict'])
    root_router.eval()
    logger.info(f"Successfully loaded checkpoint: {args.checkpoint}")

    # =========================================================
    # RECURSIVE ROUTER INTERCEPTION
    # =========================================================
    router_hooks = []
    router_probs = {}
    router_branch_names = {}

    def attach_hooks(node):
        """Walks the MoE tree and attaches a listener to every Router gate."""
        if isinstance(node, HmoeRouter):
            router_branch_names[node.name] = [child.name for child in node.branches]
            
            # --- THE FIX: Only attach hook if the gate physically exists (Ignore PASS_THROUGH) ---
            if getattr(node, 'gate', None) is not None:
                def make_hook(name):
                    def hook(module, input, output):
                        # Store output shape: [Seq, Num_Branches] for this specific router
                        router_probs[name] = output.detach().cpu().numpy()[0] 
                    return hook
                
                handle = node.gate.register_forward_hook(make_hook(node.name))
                router_hooks.append(handle)
            else:
                # If gate is None (PASS_THROUGH), mock 100% traffic to all branches for the dashboard
                router_probs[node.name] = np.ones((sequence_length, len(node.branches)))
            
            for child in node.branches:
                attach_hooks(child)

    attach_hooks(root_router)

    # ---------------------------------------------------------
    # THE BASELINE FORWARD PASS
    # ---------------------------------------------------------
    logger.info("Executing baseline inference...")
    with torch.no_grad():
        payload = HmoeInput(tensor=clean_master)
        baseline_preds = root_router(payload)

    # Cleanup Hooks
    for handle in router_hooks:
        handle.remove()

    # --- Safe Extraction for Baseline Probabilities ---
    task_top_config = global_tasks.get('task_top')
    if task_top_config and getattr(task_top_config, 'enabled', True) and 'task_top' in baseline_preds.task_logits:
        prob_top = F.softmax(baseline_preds.task_logits['task_top'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
    else:
        prob_top = np.zeros(sequence_length)

    task_bot_config = global_tasks.get('task_bot')
    if task_bot_config and getattr(task_bot_config, 'enabled', True) and 'task_bot' in baseline_preds.task_logits:
        prob_bot = F.softmax(baseline_preds.task_logits['task_bot'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
    else:
        prob_bot = np.zeros(sequence_length)
        
    # --- ADDED: Extraction for task_bull ---
    task_bull_config = global_tasks.get('task_bull')
    if task_bull_config and getattr(task_bull_config, 'enabled', True) and 'task_bull' in baseline_preds.task_logits:
        prob_bull = F.softmax(baseline_preds.task_logits['task_bull'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
    else:
        prob_bull = np.zeros(sequence_length)

    # ---------------------------------------------------------
    # STRICT SEQUENTIAL OCCLUSION
    # ---------------------------------------------------------
    logger.info("Executing Feature Occlusion Analysis...")
    # Add 'bull' to the impacts dictionary
    feature_impacts = {f.name: {'top': np.zeros(sequence_length), 'bot': np.zeros(sequence_length), 'bull': np.zeros(sequence_length)} 
                       for f in clean_master.indices}

    with torch.no_grad():
        for f_idx, feature in enumerate(clean_master.indices):
            occluded_math = clean_master.tensor.clone()
            occluded_math[:, :, f_idx] = 0.0 
            
            occ_tensor = HmoeTensor(tensor=occluded_math, indices=clean_master.indices)
            occ_payload = HmoeInput(tensor=occ_tensor)
            occ_preds = root_router(occ_payload)
            
            if task_top_config and getattr(task_top_config, 'enabled', True) and 'task_top' in occ_preds.task_logits:
                occ_prob_top = F.softmax(occ_preds.task_logits['task_top'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
            else:
                occ_prob_top = np.zeros(sequence_length)
                
            if task_bot_config and getattr(task_bot_config, 'enabled', True) and 'task_bot' in occ_preds.task_logits:
                occ_prob_bot = F.softmax(occ_preds.task_logits['task_bot'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
            else:
                occ_prob_bot = np.zeros(sequence_length)
                
            if task_bull_config and getattr(task_bull_config, 'enabled', True) and 'task_bull' in occ_preds.task_logits:
                occ_prob_bull = F.softmax(occ_preds.task_logits['task_bull'].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
            else:
                occ_prob_bull = np.zeros(sequence_length)
            
            feature_impacts[feature.name]['top'] = np.abs(prob_top - occ_prob_top)
            feature_impacts[feature.name]['bot'] = np.abs(prob_bot - occ_prob_bot)
            feature_impacts[feature.name]['bull'] = np.abs(prob_bull - occ_prob_bull)

    # ---------------------------------------------------------
    # DASHBOARD & HOVER TEXT ASSEMBLY
    # ---------------------------------------------------------
    hover_texts = []
    for i in range(sequence_length):
        txt = f"<b>Multi-Task Profile:</b><br>"
        txt += f" - Top Prob: {prob_top[i]:.3f}<br>"
        txt += f" - Bot Prob: {prob_bot[i]:.3f}<br>"
        txt += f" - Bull Prob: {prob_bull[i]:.3f}<br><br>"
        
        # --- Nested Router Display ---
        txt += f"<b>Router Delegation:</b><br>"
        for r_name, b_names in router_branch_names.items():
            if r_name in router_probs:
                probs = router_probs[r_name][i]
                txt += f" <i>{r_name}</i><br>"
                for b_idx, b_name in enumerate(b_names):
                    if b_idx < len(probs) and probs[b_idx] > 0.0001: 
                        txt += f"  ├─ {b_name}: {probs[b_idx]:.2%}<br>"
        txt += "<br>"

        step_top_impacts = sorted([(k, v['top'][i]) for k, v in feature_impacts.items()], key=lambda x: x[1], reverse=True)
        step_bot_impacts = sorted([(k, v['bot'][i]) for k, v in feature_impacts.items()], key=lambda x: x[1], reverse=True)
        step_bull_impacts = sorted([(k, v['bull'][i]) for k, v in feature_impacts.items()], key=lambda x: x[1], reverse=True)
        
        total_top = sum(x[1] for x in step_top_impacts) + 1e-9
        total_bot = sum(x[1] for x in step_bot_impacts) + 1e-9
        total_bull = sum(x[1] for x in step_bull_impacts) + 1e-9

        # --- Micro-Percentage Feature Display ---
        txt += f"<b>Impact on TOP Decision:</b><br>"
        for k, v in step_top_impacts: 
            pct = (v / total_top) * 100
            if pct >= 0.01: 
                txt += f"{k}: <b>{pct:.2f}%</b><br>"
                
        txt += f"<br><b>Impact on BOT Decision:</b><br>"
        for k, v in step_bot_impacts: 
            pct = (v / total_bot) * 100
            if pct >= 0.01: 
                txt += f"{k}: <b>{pct:.2f}%</b><br>"
                
        txt += f"<br><b>Impact on BULL Decision:</b><br>"
        for k, v in step_bull_impacts: 
            pct = (v / total_bull) * 100
            if pct >= 0.01: 
                txt += f"{k}: <b>{pct:.2f}%</b><br>"
                
        hover_texts.append(txt)

    # =========================================================================
    # PLOTLY CONSTRUCTION
    # =========================================================================
    num_indicators = len(input_feature_names)
    total_rows = 4 + num_indicators
    
    # Compute dynamic row heights: Price gets 4 units, core probs 1 unit each.
    row_heights = [4.0, 1.0, 1.0, 1.0] + [1.25] * num_indicators
    
    fig = make_subplots(
        rows=total_rows, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.01, 
        row_heights=row_heights
    )
    
    x_indices = np.arange(sequence_length)
    offset_dist = np.nanmean(ohlcv_dict["close"]) * 0.0015

    # 1. Price Chart
    fig.add_trace(go.Candlestick(x=x_indices, open=ohlcv_dict["open"], high=ohlcv_dict["high"], low=ohlcv_dict["low"], close=ohlcv_dict["close"], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    # 2. Extract Raw Targets & Pre-compute arrays for the indicator panels
    actual_top_idx = []
    if task_top_config and getattr(task_top_config, 'label_target', None):
        top_label_key = task_top_config.label_target.name
        if top_label_key in raw_data:
            actual_top_idx = np.where(np.array(raw_data[top_label_key]) == 1.0)[0]
            fig.add_trace(go.Scatter(x=actual_top_idx, y=ohlcv_dict["high"][actual_top_idx] + offset_dist, mode='markers', marker=dict(symbol='star', color='orange', size=12, line=dict(width=1, color='black')), name='Actual Tops'), row=1, col=1)
            
    actual_bot_idx = []
    if task_bot_config and getattr(task_bot_config, 'label_target', None):
        bot_label_key = task_bot_config.label_target.name
        if bot_label_key in raw_data:
            actual_bot_idx = np.where(np.array(raw_data[bot_label_key]) == -1.0)[0]
            fig.add_trace(go.Scatter(x=actual_bot_idx, y=ohlcv_dict["low"][actual_bot_idx] - offset_dist, mode='markers', marker=dict(symbol='star', color='yellow', size=12, line=dict(width=1, color='black')), name='Actual Bottoms'), row=1, col=1)

    # 3. Model Signals & Regime-Conditioned Execution
    dynamic_buy_threshold = args.buy_threshold + ((1.0 - prob_bull) * args.bear_penalty)
       
    fire_top_idx = np.where(prob_top >= args.sell_threshold)[0]
    fire_bot_idx = np.where(prob_bot >= dynamic_buy_threshold)[0]

    fig.add_trace(go.Scatter(x=fire_bot_idx, y=ohlcv_dict["low"][fire_bot_idx] - (offset_dist * 2), mode='markers', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=1, color='black')), name='Model BUY Signal', text=[hover_texts[i] for i in fire_bot_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fire_top_idx, y=ohlcv_dict["high"][fire_top_idx] + (offset_dist * 2), mode='markers', marker=dict(symbol='triangle-down', color='red', size=14, line=dict(width=1, color='black')), name='Model SELL Signal', text=[hover_texts[i] for i in fire_top_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # 4. Probability Oscillators
    fig.add_trace(go.Scatter(x=x_indices, y=prob_top, mode='lines', name='Prob: Top', line=dict(color='red', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=args.sell_threshold, line_dash="dash", line_color="white", row=2, col=1, opacity=0.5)

    fig.add_trace(go.Scatter(x=x_indices, y=prob_bot, mode='lines', name='Prob: Bottom', line=dict(color='cyan', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=3, col=1)
    
    # --- UPGRADED: Plot the dynamic threshold ---
    fig.add_trace(go.Scatter(x=x_indices, y=dynamic_buy_threshold, mode='lines', name='Dynamic Buy Threshold', line=dict(color='white', width=1, dash='dash'), opacity=0.5), row=3, col=1)

    # --- Bull Market Probability Oscillator ---
    fig.add_trace(go.Scatter(x=x_indices, y=prob_bull, mode='lines', name='Prob: Bull Market', line=dict(color='green', width=1.5), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)', text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=4, col=1)
    
    # Optional: Plot the "ground truth" bull market as a background fill on row 4
    if task_bull_config and getattr(task_bull_config, 'label_target', None):
        bull_label_key = task_bull_config.label_target.name
        if bull_label_key in raw_data:
            actual_bull_signal = np.array(raw_data[bull_label_key])
            fig.add_trace(go.Scatter(x=x_indices, y=actual_bull_signal, mode='lines', name='Actual Bull Regime', line=dict(color='white', width=1, dash='dot'), opacity=0.3), row=4, col=1)

    # =========================================================================
    # 5. SANITIZED INPUT FEATURE PANELS
    # =========================================================================
    for i, f_name in enumerate(input_feature_names):
        row_idx = 5 + i
        if f_name in feature_to_idx:
            f_data = master_np[:, feature_to_idx[f_name]]
            
            # Base indicator line
            fig.add_trace(
                go.Scatter(
                    x=x_indices, 
                    y=f_data, 
                    mode='lines', 
                    name=f_name, 
                    line=dict(width=1),
                    hovertemplate="%{y:.4f}<extra></extra>"
                ), 
                row=row_idx, 
                col=1
            )
            
            # --- OVERLAY: Ground Truth Targets (Yellow/Orange Stars) ---
            if len(actual_top_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=actual_top_idx, 
                        y=f_data[actual_top_idx], 
                        mode='markers', 
                        marker=dict(symbol='star', color='orange', size=8, line=dict(width=1, color='black')), 
                        showlegend=False, 
                        name='Actual Top Value',
                        hovertemplate="Top Marker Value: %{y:.4f}<extra></extra>"
                    ), 
                    row=row_idx, 
                    col=1
                )
                
            if len(actual_bot_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=actual_bot_idx, 
                        y=f_data[actual_bot_idx], 
                        mode='markers', 
                        marker=dict(symbol='star', color='yellow', size=8, line=dict(width=1, color='black')), 
                        showlegend=False, 
                        name='Actual Bot Value',
                        hovertemplate="Bot Marker Value: %{y:.4f}<extra></extra>"
                    ), 
                    row=row_idx, 
                    col=1
                )

            # --- OVERLAY: Model Output Hits (Cyan/Red Highlighters) ---
            if len(fire_top_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fire_top_idx, 
                        y=f_data[fire_top_idx], 
                        mode='markers', 
                        marker=dict(color='red', size=6, opacity=0.5), 
                        showlegend=False, 
                        hoverinfo='skip'  # Skip hover so it doesn't clutter the tooltip
                    ), 
                    row=row_idx, 
                    col=1
                )
                
            if len(fire_bot_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fire_bot_idx, 
                        y=f_data[fire_bot_idx], 
                        mode='markers', 
                        marker=dict(color='cyan', size=6, opacity=0.5), 
                        showlegend=False, 
                        hoverinfo='skip'
                    ), 
                    row=row_idx, 
                    col=1
                )
            
            # Add a subtle zero-line
            fig.add_hline(y=0, line_dash="dot", line_color="white", row=row_idx, col=1, opacity=0.2)
            
            # --- In-chart Labeling ---
            fig.add_annotation(
                x=0.01, 
                y=0.95, 
                xref=f"x{row_idx} domain", 
                yref=f"y{row_idx} domain",
                text=f"<b>{f_name}</b>", 
                showarrow=False, 
                font=dict(color="white", size=11),
                bgcolor="rgba(0, 0, 0, 0.6)", 
                borderpad=3,
                xanchor="left", 
                yanchor="top"
            )

    # Dynamically scale the UI height based on the number of features attached
    calc_height = 1000 + (200 * num_indicators)
    title_str = f"HMoE2 Engine | SELL: {args.sell_threshold:.2f} | BASE BUY: {args.buy_threshold:.2f} | MAX BEAR PENALTY: +{args.bear_penalty:.2f}"
    
    fig.update_layout(
        title=title_str, 
        template="plotly_dark", 
        hovermode="x unified", 
        height=calc_height, 
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.show()

if __name__ == "__main__":
    main()