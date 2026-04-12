# HMoE2 Feature Normalization Guide (`HmoeSanitizer`)

In quantitative machine learning, providing raw data to a Mixture of Experts (MoE) or Neural Network is a recipe for **Feature Dominance** (where the network only listens to the feature with the largest absolute values) and **Black Swan Vulnerability** (where market crashes blow up the network's weights). 

The `HmoeSanitizer` enforces stationary, scaled, and bounded inputs. This document deeply explains the 6 normalization types available, what they do mathematically, and when to use them.

---

## TYPE 1: Static Zero-Centered Scale 
**Mathematical Operation:** `(x - 50.0) / 50.0`
**Output Range:** `[-1, 1]`

### Deep Dive
This is a static, hardcoded transformation for indicators that are already strictly bounded between 0 and 100. It shifts the neutral midpoint (50) to `0.0` and scales the extremes (0 and 100) to `-1.0` and `+1.0`. **It does not use rolling averages**, preserving the absolute historical meaning of the value.

### Financial Example: Relative Strength Index (RSI)
If you use a rolling scaler on RSI, a prolonged bull market where RSI averages 65 will make 65 the new "zero". The network would think momentum is neutral when it is actually heavily overbought. By using Type 1, an RSI of 50 is always `0.0`, an RSI of 80 is always `+0.6`, and an RSI of 20 is always `-0.6`. The absolute meaning of "overbought" and "oversold" is preserved across decades of data.

---

## TYPE 2: Dynamic Rolling Z-Score
**Mathematical Operation:** `(x - rolling_mean) / rolling_std`
**Output Range:** Unbounded (typically `[-3, 3]`)

### Deep Dive
This transforms data into standard deviations from the recent mean. It constantly shifts the "zero" line to match the average of the rolling window. This makes non-stationary data stationary, but it destroys absolute reference points.

### Financial Example: Relative Trading Volume
Raw volume is meaningless across different eras (e.g., AAPL volume in 2005 vs 2025). By applying a rolling Z-score, you tell the network: *"Today's volume is 2.5 standard deviations higher than the average of the last 20 days."* This gives the network a perfectly stationary signal of "high activity" vs "low activity", regardless of the underlying raw share count.

---

## TYPE 3: Rolling MinMax Scaling
**Mathematical Operation:** `(x - rolling_min) / (rolling_max - rolling_min)`
**Output Range:** strictly `[0, 1]`

### Deep Dive
This scales the current value based on the highest high and lowest low within the rolling window. It is highly sensitive to outliers, as a single massive spike will widen the denominator and squash all other normal values close to zero.

### Financial Example: Donchian Channel Position / Stochastic Oscillator
If you track the raw closing price, Type 3 normalization tells the network exactly where today's price sits relative to the 80-day trading range. If the value is `1.0`, we are at the 80-day high. If `0.0`, we are at the 80-day low.

---

## TYPE 4: Log-Transform
**Mathematical Operation:** `sign(x) * log(1 + abs(x))`
**Output Range:** Symmetrical, compressed (Logarithmic)

### Deep Dive
This applies a symmetrical logarithmic compression. Crucially, **it preserves `0.0` exactly**. It squashes massive exponential spikes into manageable ranges without shifting the baseline.

### Financial Example: Wide Spread Disparities
If a metric occasionally spikes from `0.5` to `500.0`, it will overwhelm the optimizer. Type 4 compresses a `+5.0` to `~1.79`, and a `-5.0` to `-1.79`. A perfectly neutral `0.0` spread yields exactly `0.0`. It controls the "volume" of wild features while honoring the true zero-crossing.

---

## TYPE 5: Robust Tanh Estimator (Mean-Centered)
**Mathematical Operation:** `tanh(0.5 * ((x - rolling_mean) / rolling_std))`
**Output Range:** strictly `[-1, 1]`

### Deep Dive
Also known as the "Shock Absorber." It is a Z-score wrapped in a hyperbolic tangent function. Under normal conditions (within 1 or 2 standard deviations), it behaves linearly. But during extreme outliers, instead of shooting to `+10` or `-15` and breaking the neural network's weights, it smoothly asymptotes against `+1.0` or `-1.0`. **Note: It shifts the zero-line based on the rolling mean.**

### Financial Example: Daily Log-Returns
During the COVID-19 crash, daily returns produced 10-sigma outlier moves. A normal Z-score (Type 2) would pass a `-10.0` into the network, blinding it for the next 80 days because the denominator explodes. Type 5 registers the crash cleanly as a `-0.99` (maximum panic), keeping the feature mathematically balanced with the rest of the dataset.

---

## TYPE 6: Robust Tanh (Zero-Anchored)
**Mathematical Operation:** `tanh(x / rolling_std)`
**Output Range:** strictly `[-1, 1]`

### Deep Dive
The ultimate weapon for absolute oscillators. It calculates the rolling standard deviation to cushion extreme outliers, but **it does NOT subtract the mean**. This guarantees that the mathematical zero in the raw data remains exactly zero in the normalized tensor. 

### Financial Example: Moving Average (SMA) Spread
If the Fast SMA is below the Slow SMA, the raw spread is negative (Downtrend). If you use a Mean-Centered scaler (Type 5), a slight bounce in a deep downtrend might result in a "positive" normalized value, causing the AI to hallucinate a Bull Regime while still underwater. Type 6 applies the Tanh compression to prevent huge trend spikes from deafening the AI, but ensures that a negative spread *always* remains a negative number. Zero remains the exact crossover point.