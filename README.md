# HMoE2: Hierarchical Mixture of Experts

## Introduction
HMoE2 is a PyTorch-based quantitative trading and time-series forecasting engine. It implements a fully modular Hierarchical Mixture of Experts (HMoE) architecture, allowing you to route complex data streams through dynamically selected, heterogeneous neural networks.

## What It Is
Unlike traditional ensembles that blend models statically, HMoE2 uses learned routing gates to conditionally activate specialized neural backends based on the current market regime. 

**Key Features:**
* **Hierarchical Topology:** Nest routers within routers to create multi-level decision trees using simple YAML configurations.
* **Heterogeneous Backends:** Hot-swap "expert brains" depending on the required memory structure. Supported engines include:
  * `LINEAR` (Zero-memory momentum/scalping)
  * `RNN` / `GRU` / `LSTM` (Short to long-term structural sequential memory)
  * `TCN` (Temporal Convolutional Networks for swing/wave analysis)
  * `TRANSFORMER` (Causal attention for distant historical fractals)
  * `GATED_RESIDUAL` (Dynamic noise suppression for volatile data)
  * `ROUGH_PATH` (Dynamic N-dimensional pattern detection)
  * `MOTIFS` (Dynamic 1-dimensional pattern detection)
* **Sparse Gating (Top-K):** Forces hyper-specialization and compute efficiency by zeroing out unselected experts and utilizing load-balancing penalties.
* **Strict Data Isolation:** Custom Data Transfer Objects (`HmoeTensor`, `HmoeInput`, `HmoeOutput`) enforce shape safety, index tracking, and feature subsetting to prevent dimension mismatch crashes.
* **Continuous Focal Loss:** A custom loss engine optimized for continuous Gaussian heatmap targets (e.g., predicting exact market capitulation points).

## Quick Setup

Clone the repository and set up your environment:

```bash
# Clone the repository
git clone [https://github.com/jpueberbach4/hmoe2.git](https://github.com/jpueberbach4/hmoe2.git)

# Navigate into the project directory
cd hmoe2

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install torch pyyaml requests
```

(Note: For GPU acceleration, ensure you install a CUDA-compatible version of PyTorch).

See support directory.

## Matrix Profiles vs. Rough Path Signatures
When configuring your experts, it is critical to understand the mathematical distinction between the MOTIFS (Matrix Profile) and SIGNATURE (Rough Path) backends.

>Matrix Profiles: Independent 1D Shapes

Matrix Profiles handle multivariate inputs by isolating each feature into its own dimension. The MotifsBackend slides a rigid, fixed-length window across the time series and calculates sliding Pearson correlations for each feature independently.

It acts as a learnable memory bank. It can learn exactly what a specific "V-shape" looks like on an RSI, and separately learn what a "W-shape" looks like on a Z-Score, but it does not mathematically combine them. If a capitulation pattern stretches beyond the fixed window size, the Matrix Profile will suffer from "amnesia" and fail to recognize it.

>Rough Path Signatures: N-Dimensional Geometry

In stark contrast to Matrix Profiles, Rough Path Signatures treat multiple features as a single, continuous, multidimensional path moving through space.

Instead of isolating features, the SignatureBackend computes the Iterated Integrals across the entire path. When fed multiple indicators simultaneously, the math automatically generates Cross-Terms. It doesn't just track what the RSI did and what the Z-Score did; it mathematically captures how the RSI moved relative to the Z-Score (e.g., the 2D geometric "area" or topological loop created by both indicators moving at the same time).

Furthermore, Rough Path Signatures solve the "Accordion Problem" because they are time-parameterization invariant:

A Matrix Profile is vulnerable to time-warping (a crash taking 5 days looks completely different to its rigid window than a crash taking 5 weeks).

A Signature measures pure geometry, outputting a consistent state "barcode" regardless of the market's velocity. It translates the slow, grinding chop of a multi-month downtrend and the sharp violence of a flash-crash into stable, recognizable macro-regimes for the downstream task heads.

## Note

I stripped down the example. Building a reliable regime detection system is extremely difficult. I’ve gone through a lot of academic research on predicting trend shifts, and for a while I thought I had it figured out. I was seeing a 66% win rate with 128R over two years—until I discovered there was data leakage.

I’ve experimented with:

- HMM
- SNN with Rough Path features
- Triple Barrier Method (various labeling approaches)
- Brute-force indicator correlations using Rough Path theory
- Top/bottom detection (surprisingly the most reliable so far)
- Indicator manifolds (even rendered them 3d (as a walkforward 3D movie))
- Various neural network architectures and combinations

The reality is: it’s just hard. The noise in the market is overwhelming. Signal to noise ratio is bad.

That said, the architecture itself is solid. Even a small amount of leakage gets picked up instantly, and the model will happily turn it into a “profitable” neural net.

If market prediction were easy, everyone would already be doing it.

I will continue digging, I know it is possible to achieve a statistical edge. 

It is very hard to consistently get numbers like these:

```bash
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | STOP & REVERSE BACKTEST RESULTS (Pure Regime Following)
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | ============================================================
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Total Trades : 89
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Win Rate     : 48.31%
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Net PnL (%)  : 32.37% (Cumulative Uncompounded)
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Ann. Sharpe  : 2.46
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Longs        : 45
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | Shorts       : 44
2026-04-12 18:25:33 | INFO    | Macro_Visualizer | ============================================================
````

One of the most important things is that your features are scaled properly and are all in the same range.

## License

This project is licensed under the MIT License.
