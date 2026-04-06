import os
import argparse
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hmoe2.nodes import HmoeNode, HmoeRouter
from hmoe2.tensor import HmoeTensor, HmoeInput
from hmoe2.sanitize import HmoeSanitizer
from hmoe2.schema import HmoeCheatFeature

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True 
)
logger = logging.getLogger("Visualizer")


@dataclass
class EngineContext:
    """Holds the initialized state, models, and tensors for the HMoE2 pipeline."""
    device: torch.device
    args: argparse.Namespace
    root_router: Any
    global_tasks: dict
    clean_master: HmoeTensor
    master_np: np.ndarray
    ohlcv_dict: dict
    raw_data: dict
    feature_to_idx: dict
    input_feature_names: list
    sequence_length: int


@dataclass
class TelemetryData:
    """Holds the extracted inference telemetry and hook activations."""
    prob_top: np.ndarray
    prob_bot: np.ndarray
    prob_bull: np.ndarray
    router_probs: dict
    feature_impacts: dict
    motif_node: Optional[Any] = None
    motif_activations: Optional[np.ndarray] = None
    motif_feature_names: List[str] = None
    sig_node: Optional[Any] = None
    sig_activations: Optional[np.ndarray] = None
    sig_feature_names: List[str] = None


def fetch_raw_api_data(
    symbol: str, 
    timeframe: str, 
    fields: List[str], 
    start_date: str, 
    end_date: str, 
    base_url: str = "http://localhost:8000"
) -> Dict[str, list]:
    """Fetches raw OHLCV and indicator data from the local API.

    Args:
        symbol (str): The ticker symbol to fetch.
        timeframe (str): The timeframe resolution (e.g., '4h', '1d').
        fields (List[str]): List of specific feature fields to request.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        base_url (str, optional): Base URL of the data API. Defaults to localhost.

    Returns:
        Dict[str, list]: A dictionary mapping feature names to their data arrays.
    """
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
    """Recursively searches the router tree for a specific backend type.

    Args:
        node (Any): The root or current node in the HMoE tree.
        target_types (List[str]): List of valid backend type strings to match.

    Returns:
        Optional[Any]: The matching backend node, or None if not found in the topology.
    """
    if getattr(node, 'core', None) and getattr(node, 'backend_type', '').upper() in target_types:
        return node
    if hasattr(node, 'branches'):
        for child in node.branches:
            result = find_target_backend(child, target_types)
            if result:
                return result
    return None


def extract_task_probabilities(preds: Any, global_tasks: dict, task_name: str, sequence_length: int) -> np.ndarray:
    """Safely extracts sigmoid probabilities for a specific task from model predictions.

    Args:
        preds (Any): The prediction output object from the HMoE router.
        global_tasks (dict): The dictionary of registered global tasks.
        task_name (str): The name of the task to extract.
        sequence_length (int): Fallback length for initializing empty arrays.

    Returns:
        np.ndarray: A 1D array of probabilities matching the sequence length.
    """
    task_config = global_tasks.get(task_name)
    if task_config and getattr(task_config, 'enabled', True) and task_name in preds.task_logits:
        return F.softmax(preds.task_logits[task_name].to_tensor(), dim=-1)[0, :, 1].cpu().numpy()
    return np.zeros(sequence_length)


def initialize_environment(args: argparse.Namespace) -> EngineContext:
    """Bootstraps the model, fetches data, and sanitizes the input pipeline.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        EngineContext: A dataclass holding the initialized state and tensors.
    """
    torch.serialization.add_safe_globals([HmoeNode])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at: {args.checkpoint}")
        sys.exit(1)

    logger.info("Initializing Graph from YAML...")
    root_router = HmoeNode.from_yaml(args.config)
    
    global_tasks = {}
    root_router._gather_tasks(global_tasks)
    
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

    required_feature_names = [f.name for f in required_feature_objects]
    raw_data = fetch_raw_api_data(args.symbol, "4h", required_feature_names, args.start, args.end)
    
    ohlcv_dict = {
        'open': np.array(raw_data['open'], dtype=float),
        'high': np.array(raw_data['high'], dtype=float),
        'low': np.array(raw_data['low'], dtype=float),
        'close': np.array(raw_data['close'], dtype=float),
    }

    dirty_tensor = HmoeTensor.from_dict(raw_data)
    
    clean_master = HmoeSanitizer.sanitize(
        raw_tensor=dirty_tensor, 
        allowed_features=required_feature_objects, 
        drop_nan_columns=True,
        verbose=True
    ).to(device)
    
    sequence_length = clean_master.tensor.size(1)

    cheat_feature_names = {f.name for f in root_router.subtree_features if isinstance(f, HmoeCheatFeature)}
    blinded_math = clean_master.tensor.clone()

    for col_idx, feature in enumerate(clean_master.indices):
        if feature.name in cheat_feature_names:
            logger.info(f"Deadbolt engaged. Blinding cheat feature: {feature.name}")
            blinded_math[:, :, col_idx] = 0.0

    clean_master = HmoeTensor(tensor=blinded_math, indices=clean_master.indices)
    master_np = clean_master.tensor[0].cpu().numpy()
    feature_to_idx = {f.name: idx for idx, f in enumerate(clean_master.indices)}
    
    input_feature_names = []
    seen_inputs = set()
    for f in root_router.subtree_features:
        if not isinstance(f, HmoeCheatFeature) and f.name not in seen_inputs:
            input_feature_names.append(f.name)
            seen_inputs.add(f.name)

    root_router.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    root_router.load_state_dict(checkpoint['model_state_dict'])
    root_router.eval()
    logger.info(f"Successfully loaded checkpoint: {args.checkpoint}")

    return EngineContext(
        device=device,
        args=args,
        root_router=root_router,
        global_tasks=global_tasks,
        clean_master=clean_master,
        master_np=master_np,
        ohlcv_dict=ohlcv_dict,
        raw_data=raw_data,
        feature_to_idx=feature_to_idx,
        input_feature_names=input_feature_names,
        sequence_length=sequence_length
    )


def execute_telemetry_pass(ctx: EngineContext) -> TelemetryData:
    """Executes a baseline forward pass while capturing internal routing and backend telemetry.

    Args:
        ctx (EngineContext): The initialized execution context.

    Returns:
        TelemetryData: A dataclass containing probabilities and intercepted network activations.
    """
    router_hooks = []
    router_probs = {}
    router_branch_names = {}

    def attach_router_hooks(node):
        if isinstance(node, HmoeRouter):
            router_branch_names[node.name] = [child.name for child in node.branches]
            if getattr(node, 'gate', None) is not None:
                def make_hook(name):
                    def hook(module, input, output):
                        router_probs[name] = output.detach().cpu().numpy()[0] 
                    return hook
                handle = node.gate.register_forward_hook(make_hook(node.name))
                router_hooks.append(handle)
            else:
                router_probs[node.name] = np.ones((ctx.sequence_length, len(node.branches)))
            for child in node.branches:
                attach_router_hooks(child)

    attach_router_hooks(ctx.root_router)

    motif_node = find_target_backend(ctx.root_router, ["MOTIFS", "MATRIX_PROFILE", "MP"])
    motif_activations = None
    motif_handle = None
    motif_feature_names = []

    if motif_node:
        motif_feature_names = [f.name for f in motif_node.features]
        def motif_hook(module, input, output):
            nonlocal motif_activations
            flat = input[0].detach().cpu().numpy()[0]
            motif_activations = flat.reshape(-1, motif_node.core.num_motifs, motif_node.core.input_dim)
        motif_handle = motif_node.core.net.register_forward_hook(motif_hook)

    sig_node = find_target_backend(ctx.root_router, ["SIGNATURE", "SIGNATORY", "ROUGH_PATH", "RP"])
    sig_activations = None
    sig_handle = None
    sig_feature_names = []

    if sig_node:
        sig_feature_names = [f.name for f in sig_node.features]
        def sig_hook(module, input, output):
            nonlocal sig_activations
            sig_activations = output[0].detach().cpu().numpy()
        sig_handle = sig_node.core.register_forward_hook(sig_hook)

    logger.info("Executing baseline inference pass...")
    with torch.no_grad():
        payload = HmoeInput(tensor=ctx.clean_master)
        baseline_preds = ctx.root_router(payload)

    for handle in router_hooks:
        handle.remove()
    if motif_handle: motif_handle.remove()
    if sig_handle: sig_handle.remove()

    prob_top = extract_task_probabilities(baseline_preds, ctx.global_tasks, 'task_top', ctx.sequence_length)
    prob_bot = extract_task_probabilities(baseline_preds, ctx.global_tasks, 'task_bot', ctx.sequence_length)
    prob_bull = extract_task_probabilities(baseline_preds, ctx.global_tasks, 'task_bull', ctx.sequence_length)

    return TelemetryData(
        prob_top=prob_top,
        prob_bot=prob_bot,
        prob_bull=prob_bull,
        router_probs=router_probs,
        feature_impacts={},
        motif_node=motif_node,
        motif_activations=motif_activations,
        motif_feature_names=motif_feature_names,
        sig_node=sig_node,
        sig_activations=sig_activations,
        sig_feature_names=sig_feature_names
    )


def compute_feature_occlusions(ctx: EngineContext, tel: TelemetryData) -> None:
    """Iteratively masks input features to measure their isolated impact on task predictions.

    Args:
        ctx (EngineContext): The execution context containing base tensors.
        tel (TelemetryData): The telemetry data containing base probabilities. Results 
                             are mutated directly into `tel.feature_impacts`.
    """
    logger.info("Executing feature occlusion analysis...")
    tel.feature_impacts = {f.name: {'top': np.zeros(ctx.sequence_length), 
                                    'bot': np.zeros(ctx.sequence_length), 
                                    'bull': np.zeros(ctx.sequence_length)} 
                           for f in ctx.clean_master.indices}

    with torch.no_grad():
        for f_idx, feature in enumerate(ctx.clean_master.indices):
            occluded_math = ctx.clean_master.tensor.clone()
            occluded_math[:, :, f_idx] = 0.0 
            
            occ_tensor = HmoeTensor(tensor=occluded_math, indices=ctx.clean_master.indices)
            occ_payload = HmoeInput(tensor=occ_tensor)
            occ_preds = ctx.root_router(occ_payload)
            
            occ_prob_top = extract_task_probabilities(occ_preds, ctx.global_tasks, 'task_top', ctx.sequence_length)
            occ_prob_bot = extract_task_probabilities(occ_preds, ctx.global_tasks, 'task_bot', ctx.sequence_length)
            occ_prob_bull = extract_task_probabilities(occ_preds, ctx.global_tasks, 'task_bull', ctx.sequence_length)
            
            tel.feature_impacts[feature.name]['top'] = np.abs(tel.prob_top - occ_prob_top)
            tel.feature_impacts[feature.name]['bot'] = np.abs(tel.prob_bot - occ_prob_bot)
            tel.feature_impacts[feature.name]['bull'] = np.abs(tel.prob_bull - occ_prob_bull)


def build_dashboard_tooltips(ctx: EngineContext, tel: TelemetryData, sorted_motif_indices: list) -> List[str]:
    """Generates the rich HTML hover text payloads for the dashboard points.

    Args:
        ctx (EngineContext): Pipeline context variables.
        tel (TelemetryData): Derived telemetry and probabilities.
        sorted_motif_indices (list): Ranked list of active motifs.

    Returns:
        List[str]: A sequence of HTML strings aligned with the time series index.
    """
    hover_texts = []
    
    # Pre-extract values to avoid heavy lookups during the loop
    num_motifs = tel.motif_node.core.num_motifs if tel.motif_node else 0
    num_m_features = tel.motif_node.core.input_dim if tel.motif_node else 0

    for i in range(ctx.sequence_length):
        txt = f"<b>Multi-Task Profile:</b><br>"
        txt += f" - Top Prob: {tel.prob_top[i]:.3f}<br>"
        txt += f" - Bot Prob: {tel.prob_bot[i]:.3f}<br>"
        txt += f" - Bull Prob: {tel.prob_bull[i]:.3f}<br><br>"
        
        txt += f"<b>Router Delegation:</b><br>"
        for r_name, probs in tel.router_probs.items():
            txt += f" <i>{r_name}</i><br>"
            node = find_target_backend(ctx.root_router, []) # Dummy search just to traverse, logic here simplified
        txt += "<br>"

        if tel.motif_activations is not None:
            txt += f"<b>Motif Locks (Correlation > 80%):</b><br>"
            active_motifs = []
            for m in range(num_motifs):
                for f_idx in range(num_m_features):
                    val = tel.motif_activations[i, m, f_idx]
                    # Convert activation power back to readable pearson correlation
                    if val > 0.4:
                        true_corr = np.power(val, 0.25)
                        rank = sorted_motif_indices.index(m) + 1
                        active_motifs.append(f"Rank {rank} (M{m}) on {tel.motif_feature_names[f_idx]}: {true_corr:.1%}")
            
            if active_motifs:
                for am in active_motifs: txt += f" 🎯 {am}<br>"
            else:
                txt += " <i>No strict geometric lock</i><br>"
            txt += "<br>"

        if tel.sig_activations is not None:
            sig_norm = np.linalg.norm(tel.sig_activations[i])
            txt += f"<b>Signature Activity (L2 Norm):</b> {sig_norm:.2f}<br><br>"

        step_top_impacts = sorted([(k, v['top'][i]) for k, v in tel.feature_impacts.items()], key=lambda x: x[1], reverse=True)
        step_bot_impacts = sorted([(k, v['bot'][i]) for k, v in tel.feature_impacts.items()], key=lambda x: x[1], reverse=True)
        
        total_top = sum(x[1] for x in step_top_impacts) + 1e-9
        total_bot = sum(x[1] for x in step_bot_impacts) + 1e-9

        txt += f"<b>Impact on TOP Decision:</b><br>"
        for k, v in step_top_impacts: 
            pct = (v / total_top) * 100
            if pct >= 0.01: txt += f"{k}: <b>{pct:.2f}%</b><br>"
                
        txt += f"<br><b>Impact on BOT Decision:</b><br>"
        for k, v in step_bot_impacts: 
            pct = (v / total_bot) * 100
            if pct >= 0.01: txt += f"{k}: <b>{pct:.2f}%</b><br>"
                
        hover_texts.append(txt)
    return hover_texts


def render_interactive_dashboard(ctx: EngineContext, tel: TelemetryData, hover_texts: List[str]) -> None:
    """Constructs and renders the dynamic Plotly diagnostic interface.

    Args:
        ctx (EngineContext): Pipeline context configurations.
        tel (TelemetryData): Inference telemetry matrices.
        hover_texts (List[str]): Formatted tooltip overlays.
    """
    has_motifs = tel.motif_node is not None
    has_sig = tel.sig_node is not None
    num_indicators = len(ctx.input_feature_names)
    
    motif_matches = np.zeros(tel.motif_node.core.num_motifs if has_motifs else 0, dtype=int)
    sorted_motif_indices = []
    
    if has_motifs and tel.motif_activations is not None:
        for m in range(tel.motif_node.core.num_motifs):
            motif_matches[m] = np.sum(tel.motif_activations[:, m, 0] > 0.4)
        sorted_motif_indices = np.argsort(motif_matches)[::-1].tolist()

    total_rows = 4
    row_heights = [4.0, 1.0, 1.0, 1.0]

    for f_name in ctx.input_feature_names:
        total_rows += 1
        row_heights.append(1.25)
        if has_motifs and f_name in tel.motif_feature_names:
            total_rows += 1
            row_heights.append(1.5) 

    if has_sig:
        total_rows += 1
        row_heights.append(1.5) 

    if has_motifs:
        total_rows += 1
        row_heights.append(2.5) 
    
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=row_heights)
    x_indices = np.arange(ctx.sequence_length)
    offset_dist = np.nanmean(ctx.ohlcv_dict["close"]) * 0.0015

    fig.add_trace(go.Candlestick(x=x_indices, open=ctx.ohlcv_dict["open"], high=ctx.ohlcv_dict["high"], low=ctx.ohlcv_dict["low"], close=ctx.ohlcv_dict["close"], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    actual_bot_idx = []
    task_bot_config = ctx.global_tasks.get('task_bot')
    if task_bot_config and getattr(task_bot_config, 'label_target', None):
        bot_label_key = task_bot_config.label_target.name
        if bot_label_key in ctx.raw_data:
            actual_bot_idx = np.where(np.array(ctx.raw_data[bot_label_key]) == -1.0)[0]
            fig.add_trace(go.Scatter(x=actual_bot_idx, y=ctx.ohlcv_dict["low"][actual_bot_idx] - offset_dist, mode='markers', marker=dict(symbol='star', color='yellow', size=12, line=dict(width=1, color='black')), name='Actual Bottoms'), row=1, col=1)

    dynamic_buy_threshold = ctx.args.buy_threshold + ((1.0 - tel.prob_bull) * ctx.args.bear_penalty)
    fire_bot_idx = np.where(tel.prob_bot >= dynamic_buy_threshold)[0]

    fig.add_trace(go.Scatter(x=fire_bot_idx, y=ctx.ohlcv_dict["low"][fire_bot_idx] - (offset_dist * 2), mode='markers', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=1, color='black')), name='Model BUY Signal', text=[hover_texts[i] for i in fire_bot_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x_indices, y=tel.prob_top, mode='lines', name='Prob: Top', line=dict(color='red', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=ctx.args.sell_threshold, line_dash="dash", line_color="white", row=2, col=1, opacity=0.5)

    fig.add_trace(go.Scatter(x=x_indices, y=tel.prob_bot, mode='lines', name='Prob: Bottom', line=dict(color='cyan', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_indices, y=dynamic_buy_threshold, mode='lines', name='Dynamic Buy Threshold', line=dict(color='white', width=1, dash='dash'), opacity=0.5), row=3, col=1)

    fig.add_trace(go.Scatter(x=x_indices, y=tel.prob_bull, mode='lines', name='Prob: Bull Market', line=dict(color='green', width=1.5), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)', text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=4, col=1)
    
    current_row_idx = 4
    for f_name in ctx.input_feature_names:
        current_row_idx += 1
        if f_name in ctx.feature_to_idx:
            f_data = ctx.master_np[:, ctx.feature_to_idx[f_name]]
            
            fig.add_trace(go.Scatter(x=x_indices, y=f_data, mode='lines', name=f_name, line=dict(width=1), hovertemplate="%{y:.4f}<extra></extra>"), row=current_row_idx, col=1)
            
            if has_motifs and tel.motif_activations is not None and f_name in tel.motif_feature_names:
                m_f_idx = tel.motif_feature_names.index(f_name)
                top_m_idx = sorted_motif_indices[0] 
                top_m_hit_indices = np.where(tel.motif_activations[:, top_m_idx, m_f_idx] > 0.4)[0]
                
                if len(top_m_hit_indices) > 0:
                    fig.add_trace(
                        go.Scatter(x=top_m_hit_indices, y=f_data[top_m_hit_indices], mode='markers', marker=dict(symbol='circle-open', color='magenta', size=12, line_width=2), name=f"Top Pattern (M{top_m_idx})", hoverinfo='skip'), 
                        row=current_row_idx, col=1
                    )
            
            if len(actual_bot_idx) > 0:
                fig.add_trace(go.Scatter(x=actual_bot_idx, y=f_data[actual_bot_idx], mode='markers', marker=dict(symbol='star', color='yellow', size=8, line=dict(width=1, color='black')), showlegend=False, hoverinfo='skip'), row=current_row_idx, col=1)

            fig.add_hline(y=0, line_dash="dot", line_color="white", row=current_row_idx, col=1, opacity=0.2)
            fig.add_annotation(x=0.01, y=0.95, xref=f"x{current_row_idx} domain", yref=f"y{current_row_idx} domain", text=f"<b>{f_name}</b>", showarrow=False, font=dict(color="white", size=11), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

            if has_motifs and tel.motif_activations is not None and f_name in tel.motif_feature_names:
                m_f_idx = tel.motif_feature_names.index(f_name)
                current_row_idx += 1
                
                heat_z = tel.motif_activations[:, :, m_f_idx]
                heat_z_vis = np.power(np.maximum(heat_z, 0), 0.25).T 
                heat_z_vis_sorted = heat_z_vis[sorted_motif_indices, :]
                y_labels = [f"Rank {i+1} [M{m}] ({motif_matches[m]})" for i, m in enumerate(sorted_motif_indices)]
                
                fig.add_trace(go.Heatmap(z=heat_z_vis_sorted, x=x_indices, y=y_labels, colorscale="Viridis", zmin=0.0, zmax=1.0, showscale=False, hoverinfo='skip', name=f"{f_name} Sorted X-Ray"), row=current_row_idx, col=1)
                for bot_loc in actual_bot_idx:
                    fig.add_vline(x=bot_loc, line_width=1, line_dash="dot", line_color="rgba(255, 255, 0, 0.4)", row=current_row_idx, col=1)
                fig.add_annotation(x=0.01, y=0.95, xref=f"x{current_row_idx} domain", yref=f"y{current_row_idx} domain", text=f"<b>Sorted Motif Heatmap: {f_name}</b>", showarrow=False, font=dict(color="cyan", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")

    if has_sig and tel.sig_activations is not None:
        current_row_idx += 1
        
        # Isolate signal confidence by applying local normalization
        sig_z = tel.sig_activations.T
        sig_z_min = sig_z.min(axis=1, keepdims=True)
        sig_z_max = sig_z.max(axis=1, keepdims=True) + 1e-8
        sig_z_vis = (sig_z - sig_z_min) / (sig_z_max - sig_z_min)
        
        # Apply non-linear dampening
        sig_z_vis = np.power(sig_z_vis, 3.0)

        # ======================================================================
        # NEW EXPERIMENT: ACTIVATION DENSITY WITH SCHMITT TRIGGER (HYSTERESIS)
        # ======================================================================
        hidden_dim, seq_len = sig_z_vis.shape
        
        # 1. Define what constitutes an "Awake" neuron (ignoring the faint gray hum)
        awake_threshold = 0.20 
        
        # 2. Count exactly how many neurons are awake per vertical column
        awake_counts = np.sum(sig_z_vis > awake_threshold, axis=0)
        
        # 3. Calculate the Density Ratio (0.0 to 1.0)
        density_ratio = awake_counts / hidden_dim
        
        # 4. Smooth the Density Ratio to kill the flicker (20-bar window)
        import pandas as pd
        smoothed_density = pd.Series(density_ratio).ewm(span=20, adjust=False).mean().values
        
        # 5. Define DUAL Global Thresholds (The Schmitt Trigger)
        # High water mark to declare chaos, Low water mark to declare peace.
        chaos_threshold = np.percentile(smoothed_density, 65)
        quiet_threshold = np.percentile(smoothed_density, 35)
        
        # 6. State Machine with Hysteresis
        # Initialize based on the starting density relative to the quiet threshold
        current_regime_is_quiet = smoothed_density[0] < quiet_threshold
        regime_change_indices = []
        
        for t in range(1, seq_len):
            density = smoothed_density[t]
            
            # If we are QUIET, only flip to CHAOS if we breach the HIGH threshold
            if current_regime_is_quiet and density > chaos_threshold:
                current_regime_is_quiet = False
                regime_change_indices.append((t, current_regime_is_quiet))
                
            # If we are CHAOTIC, only flip to QUIET if we drop below the LOW threshold
            elif not current_regime_is_quiet and density < quiet_threshold:
                current_regime_is_quiet = True
                regime_change_indices.append((t, current_regime_is_quiet))

        # 7. Draw the Regime Blocks on the UI
        last_start = 0 if smoothed_density[0] < quiet_threshold else None
        
        for idx, is_quiet in regime_change_indices:
            if is_quiet:
                # Regime Started (Density dropped below low threshold -> Quiet/Trending)
                last_start = idx
                
                # Draw solid green start line on both main chart and X-ray
                fig.add_vline(x=idx, line_width=2, line_dash="solid", line_color="rgba(0, 255, 0, 0.8)", row=1, col=1)
                fig.add_vline(x=idx, line_width=2, line_dash="solid", line_color="rgba(0, 255, 0, 0.8)", row=current_row_idx, col=1)
            else:
                # Regime Ended (Density spiked above high threshold -> Chaos/Reversal)
                if last_start is not None:
                    # Paint the background of the quiet regime green
                    fig.add_vrect(x0=last_start, x1=idx, fillcolor="rgba(0, 255, 0, 0.08)", layer="below", line_width=0, row=1, col=1)
                    fig.add_vrect(x0=last_start, x1=idx, fillcolor="rgba(0, 255, 0, 0.15)", layer="below", line_width=0, row=current_row_idx, col=1)
                
                # Draw dashed red end line on both main chart and X-ray
                fig.add_vline(x=idx, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)", row=1, col=1)
                fig.add_vline(x=idx, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)", row=current_row_idx, col=1)
                
                last_start = None

        # Close the final block if it ends in a quiet regime
        if last_start is not None:
            fig.add_vrect(x0=last_start, x1=seq_len, fillcolor="rgba(0, 255, 0, 0.08)", layer="below", line_width=0, row=1, col=1)
            fig.add_vrect(x0=last_start, x1=seq_len, fillcolor="rgba(0, 255, 0, 0.15)", layer="below", line_width=0, row=current_row_idx, col=1)
        # ======================================================================

        # Define custom high-contrast colorscale
        replicated_colorscale = [
            [0.0, "rgb(0, 0, 0)"],         
            [0.3, "rgb(0, 0, 0)"],         
            [0.5, "rgb(50, 20, 0)"],       
            [0.75, "rgb(200, 120, 0)"],    
            [0.9, "rgb(255, 215, 0)"],     
            [1.0, "rgb(255, 255, 200)"]    
        ]
        
        fig.add_trace(
            go.Heatmap(
                z=sig_z_vis, x=x_indices, y=[f"H{h}" for h in range(tel.sig_activations.shape[1])],
                colorscale=replicated_colorscale, 
                showscale=False, hoverinfo='skip', name="Signature X-Ray"
            ), row=current_row_idx, col=1
        )
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{current_row_idx} domain", yref=f"y{current_row_idx} domain", text=f"<b>Signature Geometry X-Ray ({', '.join(tel.sig_feature_names)})</b>", showarrow=False, font=dict(color="cyan", size=10), bgcolor="rgba(0, 0, 0, 0.6)", borderpad=3, xanchor="left", yanchor="top")  
    
    if has_motifs:
        motif_row = current_row_idx + 1
        learned_motifs_np = tel.motif_node.core.motifs.detach().cpu().numpy()
        spacing = ctx.sequence_length // max(1, tel.motif_node.core.num_motifs)
        
        for rank, m_idx in enumerate(sorted_motif_indices):
            x_offset = rank * spacing 
            
            # Artificially expand the X-axis rendering scale to make motif shapes highly visible
            x_plot = np.linspace(x_offset, x_offset + (spacing * 0.75), tel.motif_node.core.motif_length)
            shape_to_plot = learned_motifs_np[m_idx, 0, :]
            
            shape_mean = np.mean(shape_to_plot)
            shape_std = np.std(shape_to_plot) + 1e-8
            shape_to_plot = (shape_to_plot - shape_mean) / shape_std
            
            fig.add_trace(go.Scatter(x=x_plot, y=shape_to_plot, mode='lines+markers', name=f'Rank {rank+1}: M{m_idx} ({motif_matches[m_idx]} hits)', line=dict(width=3)), row=motif_row, col=1)
            
        fig.add_annotation(x=0.01, y=0.95, xref=f"x{motif_row} domain", yref=f"y{motif_row} domain", text=f"<b>AI Memory (Ranked by Frequency, Left=Highest)</b>", showarrow=False, font=dict(color="cyan", size=12), bgcolor="rgba(0, 0, 0, 0.6)")

    calc_height = 800 + (160 * num_indicators) + (350 if has_motifs else 0) + (250 if has_sig else 0)
    title_str = f"HMoE2 Engine | SELL: {ctx.args.sell_threshold:.2f} | BASE BUY: {ctx.args.buy_threshold:.2f} | MAX BEAR PENALTY: +{ctx.args.bear_penalty:.2f}"
    
    fig.update_layout(title=title_str, template="plotly_dark", hovermode="x unified", height=calc_height, xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=60, b=40))
    fig.show()


def main():
    """Main execution entry point for the HMoE2 inference visualizer."""
    parser = argparse.ArgumentParser(description="HMoE2 Strict Inference Visualizer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML topology.")
    parser.add_argument("--checkpoint", type=str, default="experiment_01_gbpusd/best_checkpoint.pt", help="Path to the .pt checkpoint file.")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-06-01", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="GBP-USD", help="Ticker Symbol")
    parser.add_argument("--sell-threshold", type=float, default=0.5, help="Override SELL threshold.")
    parser.add_argument("--buy-threshold", type=float, default=0.5, help="Override BUY threshold.")
    parser.add_argument("--bear-penalty", type=float, default=0.0, help="Max threshold penalty during bear markets.")
    args = parser.parse_args()

    # 1. State Initialization
    ctx = initialize_environment(args)

    # 2. Model Telemetry & Baseline Forward Pass
    tel = execute_telemetry_pass(ctx)

    # 3. Dynamic Feature Sensitivity Analysis
    compute_feature_occlusions(ctx, tel)

    # 4. View Rendering
    sorted_motif_indices = []
    if tel.motif_activations is not None:
        motif_matches = np.zeros(tel.motif_node.core.num_motifs, dtype=int)
        for m in range(tel.motif_node.core.num_motifs):
            motif_matches[m] = np.sum(tel.motif_activations[:, m, 0] > 0.4)
        sorted_motif_indices = np.argsort(motif_matches)[::-1].tolist()
        
    hover_texts = build_dashboard_tooltips(ctx, tel, sorted_motif_indices)
    render_interactive_dashboard(ctx, tel, hover_texts)


if __name__ == "__main__":
    main()