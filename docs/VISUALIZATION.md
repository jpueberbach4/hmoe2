# Visualization & Auditability Guide

Welcome to the visualization guidelines for this repository. 

Building quantitative models—especially complex neural architectures like Hierarchical Mixtures of Experts (HMoE) or deep sequential networks—often feels like wrestling with a black box. If your model outputs a `1.0` (Buy) or a `0.0` (Sell), knowing *that* it made a decision is useless unless you know *why* it made that decision. 

This guide outlines how to pry open the neural network, extract its internal logic via the HMoE2 API, and visualize it alongside price action without destroying your RAM.

---

## 1. Why Hooking into Neurons Matters (The "Glass-Box" Paradigm)

In standard deep learning (like image classification), a 95% accuracy rate is cause for celebration. In algorithmic trading, a 95% accuracy rate with a 5% catastrophic failure rate will liquidate your portfolio. 

We cannot rely purely on aggregate metrics like `val_loss` or `F1-score`. We must audit the network's structural understanding of the market. To do this, we use **PyTorch Hooks** to intercept the hidden activations of specific `HmoeExpert` backends *before* they are compressed by the task heads into a final prediction.

**Why do this?**
* **Detecting Catastrophic Interference:** Are your neurons firing wildly during sideways chop, or are they holding steady in a macro block?
* **Latent Velocity:** By measuring how rapidly the hidden state is changing, you can detect structural market shifts *before* the final output layer officially flips its prediction.
* **Feature Attribution:** If a specific expert suddenly demands 100% of the routing weight, inspecting its internal neurons allows you to trace that decision back to the specific indicator that triggered it.

---

## 2. Intercepting the HMoE Architecture (The Technical How-To)

Because the HMoE2 architecture is a dynamic, recursive tree loaded from YAML, you cannot simply hardcode `model.layer1`. You must dynamically walk the tree, locate the specific expert or task head you want to audit, and attach a PyTorch hook to its `core` (the backend neural network).

### Example: Loading a Checkpoint and Capturing a Hidden State

Here is how you initialize the HMoE root, load your trained weights, locate the expert responsible for the macro regime, and intercept its geometric "thoughts" during a validation pass.

```python
import torch
import numpy as np
from hmoe2.nodes import HmoeNode
from hmoe2.tensor import HmoeTensor, HmoeInput

# 1. Initialize the architecture, load the checkpoint, and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_router = HmoeNode.from_yaml("config.yaml").to(device)

# Load the trained weights
checkpoint = torch.load("experiment_01_gbpusd/best_checkpoint.pt", map_location=device, weights_only=False)
root_router.load_state_dict(checkpoint['model_state_dict'], strict=False)
root_router.eval()  # Critical: Disable dropout/batchnorm for visualization

# Prepare the hook container
captured_hidden_state = None

def core_hook(module, input_tensor, output_tensor):
    global captured_hidden_state
    # HMoE2 backends return a tensor of shape [Batch, Sequence, Hidden]
    if output_tensor.dim() == 3:
        # Detach from graph, move to CPU, grab the first sequence, and convert to NumPy
        captured_hidden_state = output_tensor[0].detach().cpu().numpy()

# 2. Walk the tree to find the expert handling a specific task
hook_handle = None
for name, module in root_router.named_modules():
    if type(module).__name__ == 'HmoeExpert' and hasattr(module, 'core'):
        # Check if this expert is assigned to the task we want to audit
        task_names = [t.name for t in getattr(module, 'allowed_tasks', [])]
        if 'task_regime' in task_names:
            print(f"Attaching Deep X-Ray Hook to: {name}")
            hook_handle = module.core.register_forward_hook(core_hook)
            break

# 3. Prepare the strictly formatted HMoE payload
# (Assuming `clean_data_dict` contains your sanitized features)
clean_master = HmoeTensor.from_dict(clean_data_dict).to(device)
payload = HmoeInput(tensor=clean_master)

# 4. Execute the forward pass
with torch.no_grad():
    predictions = root_router(payload)

# 5. Clean up the hook to prevent memory leaks
if hook_handle:
    hook_handle.remove()

# 'captured_hidden_state' now contains the raw numerical matrix of the backend's hidden state.
# You can also extract the final probabilities directly from the structured HmoeOutput:
if 'task_regime' in predictions.task_logits:
    logits = predictions.task_logits['task_regime'].to_tensor()
    probabilities = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
```

---

## 3. The "Deep X-Ray" Technique

Once you have the `captured_hidden_state` matrix (shape `[Sequence_Length, Hidden_Dimension]`), you can visualize how the expert's internal representation evolves over time. 

### Trick A: High-Contrast Neuron Heatmaps
Plotting raw activations often looks like gray mush because neural networks tend to operate in very narrow mathematical bands. To make the patterns visible to the human eye, you must aggressively normalize and contrast-enhance the matrix.

```python
# 1. Normalize each neuron (column) to a strict [0, 1] scale over time
col_mins = np.min(captured_hidden_state, axis=0, keepdims=True)
col_maxs = np.max(captured_hidden_state, axis=0, keepdims=True)
normalized_state = (captured_hidden_state - col_mins) / (col_maxs - col_mins + 1e-8)

# 2. Apply exponential contrast (e.g., power of 3) to suppress noise and highlight spikes
contrast_state = np.power(normalized_state, 3)

# 3. Transpose for plotting: Shape becomes [Neurons, Time]
heatmap_data = contrast_state.T 
```
If you plot `heatmap_data` directly below your candlestick chart, you will physically see specific horizontal bands (neurons) light up exclusively during Uptrends, while others only fire during Downtrends.

### Trick B: Latent Velocity (Advanced)
Instead of looking at the neurons directly, you can measure how violently the network's "mind" is changing. We do this by calculating the **Cosine Distance** between the current hidden state and a rolling moving average of previous hidden states.

If the market is ranging, the expert's internal state remains static (Low Velocity). The moment a true structural breakout occurs in your features, the internal state shifts dramatically to a new mathematical quadrant (High Velocity Spike).

---

## 4. UI Rendering Best Practices (Plotly WebGL)

Financial machine learning generates massive amounts of data. Rendering 5,000 candles alongside 6 indicators and a 64-dimension neuron heatmap will instantly crash a standard browser DOM.

If you are using `plotly` or similar web-based rendering tools, follow these strict rules:

1. **Never use standard Scatter:** Always use `Scattergl` (WebGL). It shifts rendering from the CPU to the GPU.
2. **Rasterize Heatmaps:** When plotting the Deep X-Ray, always pass `zsmooth='best'`. This forces the library to render the matrix as a single compressed image rather than rendering individual SVG rectangles for every single data point.
3. **Avoid Background Objects:** Do not draw thousands of individual vertical background rectangles to denote "Buy Zones". Instead, find the start and end indices of the contiguous zone, and draw a single `vrect` spanning the entire block. 

```python
import plotly.graph_objects as go

# ❌ BAD: Will crash the browser on large datasets
fig.add_trace(go.Scatter(x=timestamps, y=prices, mode='lines'))

# ✅ GOOD: Hardware accelerated
fig.add_trace(go.Scattergl(x=timestamps, y=prices, mode='lines'))

# ✅ GOOD: Heatmap optimization (Crucial for the X-Ray)
fig.add_trace(go.Heatmap(
    z=heatmap_data, 
    colorscale='inferno',
    zsmooth='best' # Forces rasterization
))
```

By combining strict hook discipline with optimized rendering, you transition from blindly trusting a loss function to actively auditing your network's structural understanding of the market.

Note: this is the CURRENT approach. Abstractions for hooks have been registered as a thing to do. It will become easier in the near-future. 