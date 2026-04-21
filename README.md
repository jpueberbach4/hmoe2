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

**UPDATE**: I’ve been working on this project for about three months. The focus has been on structuring features and experimenting with different techniques using an HMOE to achieve two objectives:

a) detect macro regimes without lookahead, and
b) identify pivots within those regimes (bottoms in uptrends, tops in downtrends), also without lookahead.

Financial markets are notoriously difficult to model due to their non-stationary nature and low signal-to-noise ratio. What I’m attempting is therefore highly challenging—especially without direct access to L2 and L3 data.

That said, certain aspects of L2 can be approximated through careful feature engineering. For example, absorption zones (buy/sell walls) can be inferred from very low time-frame price and volume data.

After many iterations, I’m starting to see stability in walk-forward regime detection, and early signs of a statistical edge are emerging. The current model, trained on GBP/USD data from 2006–2022, appears to generalize reasonably well to other assets, including Bitcoin and the Japanese yen.

Interestingly, the primary bottlenecks were not in the software, but in labeling and feature design—data leakage in particular proved to be a persistent issue. Moving to autoencoders led to a significant performance improvement, with the approach inspired by MP3 compression techniques.

I’ll be sharing details of the autoencoder architecture soon. Next, I plan to explore two additional approaches to potentially enhance performance further: Continuous Wavelet Transforms (CWT) and Singular Spectrum Analysis (SSA).

The overall architecture is now in a solid place, but the work remains experimental. Here and there additional parameters need to be inserted. Some are still hardcoded.

**Update**: Lookahead bias is still extremely tricky. Never blindly trust AI to verify scripts for it. In one case, I was tired and followed the lookahead-prevention changes suggested by AI. It completely misunderstood my setup: I use the H4 candle as the base timeframe and reconstruct information from lower timeframes (10s, 1m, and 5m) within that candle. Instead, it assumed 5m was my base timeframe.

I only caught the mistake when I explicitly asked what it thought my base timeframe was. In effect, it produced a very convincing but incorrect explanation. So be cautious when relying on AI for lookahead bias checks. (I had backups—thankfully.)

“I got completely confused by the ‘5m’ labels in your feature names, which led me to assume your model was running on a 5-minute chart. That was my mistake, not yours.

You restored your original scripts and your 0.0183 checkpoint from backups. Your ‘nice picture’ and SNN architecture are intact, causal, and ready to use.”

Moral: AI almost destroyed a perfectly fine bottom sniper model. It is so so dangerous, at times. Now this is research. Imagine working with tricky production code and having a moment of "guards down".

**Update**: If you are using this repo for financial purposes, I can give some general advice (I cant share my configs but i can share some hints) for bottom detection. Train 2006-2022. Use a multiview encoder with a 16D manifold. Train the encoder.pt, lock the weights. Configure a TCN router with hidden_dim 32, feed it the full 16D manifold as features. Below that, you configure 3 experts. A: TCN, hidden_dim 64, dilations [1, 2, 4, 8, 16, 32, 64], dropout 0.2, feed it the full 16D manifold. B: GRU, hidden_dim 64, num_layers 2, dropout 0.2, feed it the full 16D manifold. C: RP, hidden_dim 64. depth 3, dropout 0.2, feed it the full 16D manifold. Feed all the experts the same allowed_task. Do sequential runs. First the encoder, gives pt file, then feed the pt file to the trainer that uses the HMOE. Features: various features related to inter-candle strength/delta/volume (10s, 1m, 5m (5m only works too)). Labelling: only mark bottoms (window 180) (for bottom detection). For rendering. Just ask an AI and be creative. Something like: "i give you my training scripts, this creates a checkpoint file. Now visualize this checkpoint file. I want to see the ground truth, the price chart and the state of the neural net and encoder weights. Make a unified interface." If you do it right, you will get a very nice output. The multiview encoder classes are coming. I need to abstract them better first. 

## License

This project is licensed under the MIT License.
