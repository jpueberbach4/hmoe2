## What is this about

I have been actively trading for several years, experiencing alternating periods of strong performance and significant setbacks. Through this journey, it became clear that emotional decision-making was a primary weakness in my trading process. To address this, I have spent the past three months focusing on fully automating my trading workflow.

My initial effort centered on building a local data infrastructure capable of harvesting and resampling market data. The goal was to maintain complete control over the data pipeline, including the ability to define custom timeframes and operate without external rate limits. A key challenge in this phase was ensuring accurate support for custom session configurations and achieving full consistency with candle data from the Dukascopy MetaTrader 4 platform.

With unrestricted access to high-quality data, I aimed to create an environment where I could experiment extensively to identify effective automated trading strategies. Early in this process, I became convinced that the solution would involve neural networks. The challenge then shifted to determining the appropriate architectures and selecting the most relevant indicators.

The core objective has remained consistent: automatically detect the primary market trend, and within that context, identify optimal entry points—bottoms in uptrends and tops in downtrends. While this concept appears straightforward, meeting my performance standards proved significantly more complex. These standards include:

Approximately 90% accuracy in trend detection
Minimal lag during trend transitions (ideally fewer than 6 candles, or ~24 hours)
Reliable detection of sideways or “choppy” market conditions

Some may be familiar with the bp.markets.ingest project—an open-source initiative I developed to handle data ingestion, resampling, and feature generation (effectively an indicator factory), along with an API layer. Due to extensive experimentation and iterative changes, the project became increasingly difficult to maintain. I therefore chose to temporarily take it offline, with plans to either refactor it into a general-purpose library or relaunch it once it meets a higher standard of robustness and clarity.

In parallel, I explored multiple machine learning approaches, with a strong inclination toward ensemble methods composed of specialized neural networks. The idea was to develop distinct models, each excelling in a specific domain—trend detection, top identification, bottom identification, and potentially sentiment analysis.

My initial implementation, Pulsar, was a neuroevolution-based system designed to detect market bottoms. While it achieved a high F1 score, the resulting signals were too subtle and highly sensitive to changing market regimes. To address this, I developed RPulsar, a recurrent neural network variant. This improved performance further but introduced a new issue: the model attempted to learn both tops and bottoms simultaneously, leading to catastrophic interference, where learning the patterns for a top actively degraded the network's ability to remember the patterns for a bottom. 

Additionally, different indicators tend to be more effective for either tops or bottoms, yet the model was constrained to a single shared feature set. Although functional, it did not meet my quality requirements.

Given the financial implications of this work, I continued to deepen my understanding of neural network architectures and encountered the concept of Mixture of Experts (MoE). Commonly used in large-scale systems such as large language models, MoE architectures distribute learning across multiple specialized subnetworks. This aligned closely with my goal of creating expert models for distinct trading tasks.

I began experimenting with Microsoft Tutel, an MoE framework that provided promising initial results. However, it lacked the level of customization I required—particularly at the router level, which governs how inputs are distributed among experts. Additionally, its black-box nature limited transparency and control, both of which are critical for this type of system.

After gaining valuable insights from Tutel, I decided to develop a custom Mixture of Experts architecture from scratch. Unlike traditional MoE implementations focused on computational scaling across GPUs, my design emphasizes feature-space partitioning. Each expert is trained exclusively on a specific subset of features, allowing it to specialize without interference from unrelated signals.

The result is a hierarchical Mixture of Experts system in which both the router and the experts can utilize different neural network architectures, each tailored to its role. By isolating feature domains, this approach effectively eliminates catastrophic interference and enables more precise specialization.

One current limitation of this design is that experts do not share knowledge or learn from one another. This is a known trade-off, and I plan to explore solutions to enable controlled knowledge sharing in future iterations, should it prove beneficial.

## So what is the current status

The current status is that I have a functioning model for GBP/USD, achieving an average win rate of approximately 60% when entering positions on trend reversals with tight stop-losses. This figure is based on multiple backtesting periods and should be considered a consistent average rather than an isolated result.

At this stage, I am continuing to test, refine, and optimize both the underlying models and the supporting codebase.

Trend detection performance has reached a reasonably stable and satisfactory level. My current focus has shifted toward identifying intermediate market structures—specifically, detecting intermediate tops within downtrends and intermediate bottoms within uptrends.

Intermediate top detection is performing particularly well. However, bottom detection, which was previously the easier problem, is now producing overly smooth, Gaussian-like distributions. These signals lack precision and require further sharpening to improve timing and reliability.

## What is the general advice

For trend direction, I utilize two weekly indicators: RSI and MACD. Both indicators are lightly smoothed using a factor of 3. The RSI values are normalized by dividing by 100, while the MACD (specifically the MACD line) is scaled by a factor of 1000 for GBP/USD. A positive class weighting (pos_weight) between 3 and 6 is applied during training.

The model leverages the Signatory backend with a window size of 200, and the router configuration at this stage is set to a simple pass-through.

For detecting intermediate tops and bottoms, I rely on multiple RSI indicators derived from three different timeframes: 4-hour, 8-hour, and 12-hour intervals. Each RSI is configured with a period of 14 and smoothed with a factor of 5, using the same normalization approach as described above.

## Interesting

I encountered an interesting unintended result yesterday. By excessively scaling a particular feature, I effectively “distorted” the input representation of USD/JPY in a way that highlighted market turning points with remarkable precision.

This distortion acted almost like an X-ray of the price action at the exact moments where tops or bottoms occurred, producing signals that were significantly more precise than those generated by the dedicated detect_top and detect_bottom experts.

## Future

Parts of the codebase still rely on torch.tensor and need to be migrated to HMoeTensor. The HMoeTensor abstraction simplifies the mapping between features and tensor indices, improving both clarity and maintainability within the system.

The RSI-based solution will be published once I am sufficiently confident to deploy it in live trading. I intend to make it publicly available, as it is not easily arbitraged away. That said, the most recent iterations and refinements will remain private.

bp.markets.ingest will get relaunched.



