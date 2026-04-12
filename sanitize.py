import torch
import torch.nn.functional as F
import logging
from typing import List
from hmoe2.tensor import HmoeTensor
from hmoe2.schema import HmoeFeature

logger = logging.getLogger("HmoeSanitizer")
logger.propagate = False

# Configure logger only once to avoid duplicate handlers
if not logger.handlers:
    ch = logging.StreamHandler()

    # Define log message format including timestamp, level, and source
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    ch.setFormatter(formatter)

    # Attach handler and set default logging level
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class HmoeSanitizer:
    """Preprocessing utility for enforcing feature integrity and consistency.

    This class performs a sequence of transformations on input tensors,
    including feature filtering, NaN handling, rolling normalization,
    value clamping, and diagnostic logging.

    Normalization Types (set via feature.normalize integer):
        1: Static Bounded Scale (/100)
        2: Dynamic Rolling Z-Score
        3: Rolling MinMax Scaling [0, 1]
        4: Log-Transform (Volatility/Volume)
        5: Robust Tanh Estimator [-1, 1] (Outlier resistance)

    Methods:
        sanitize: Main entry point for preprocessing raw HmoeTensor data.
    """

    @staticmethod
    def sanitize(
        raw_tensor: HmoeTensor,
        allowed_features: List[HmoeFeature],
        drop_nan_columns: bool = False,
        rolling_window: int = 1080,
        verbose: bool = True
    ) -> HmoeTensor:
        """Cleans and normalizes input tensor according to feature schema."""
        
        # Clone raw tensor data to avoid mutating original input
        clean_data = raw_tensor.tensor.clone()

        # Copy feature indices for modification
        current_indices = list(raw_tensor.indices)

        # Initialize containers for filtering and feature configuration
        keep_cols = []
        keep_indices = []

        # Track feature-specific clamp and normalization settings
        active_clamps = {}
        active_norms = {}

        # Iterate through all input features and apply whitelist filtering
        for feature in current_indices:
            col_name = feature.name
            is_allowed = False

            # Check if feature matches any allowed feature definition
            for allowed_obj in allowed_features:
                if col_name == allowed_obj.name or col_name.startswith(f"{allowed_obj.name}__"):
                    is_allowed = True

                    # Capture clamp value if defined
                    if hasattr(allowed_obj, 'clamp') and allowed_obj.clamp is not None:
                        active_clamps[col_name] = allowed_obj.clamp

                    # Capture normalization flag/type if enabled
                    if hasattr(allowed_obj, 'normalize') and allowed_obj.normalize:
                        # Cast to int to maintain type 1, 2, 3, 4, or 5
                        active_norms[col_name] = int(allowed_obj.normalize)

                    break

            # Record whether column should be kept
            keep_cols.append(is_allowed)

            if is_allowed:
                # Propagate clamp configuration to feature instance if applicable
                if col_name in active_clamps and hasattr(feature, 'clamp'):
                    feature.clamp = active_clamps[col_name]

                # Propagate normalization integer to feature instance
                if col_name in active_norms and hasattr(feature, 'normalize'):
                    feature.normalize = active_norms[col_name]

                # Add feature to retained indices
                keep_indices.append(feature)

        # Apply boolean mask to filter tensor columns
        keep_tensor_mask = torch.tensor(keep_cols, dtype=torch.bool, device=clean_data.device)
        clean_data = clean_data[:, :, keep_tensor_mask]

        # Update feature indices to reflect filtered columns
        current_indices = keep_indices

        # Optionally remove columns containing any NaN values
        if drop_nan_columns:
            nan_mask = torch.isnan(clean_data).any(dim=0).any(dim=0)

            if nan_mask.any():
                keep_mask = ~nan_mask
                clean_data = clean_data[:, :, keep_mask]
                current_indices = [
                    feat for feat, keep in zip(current_indices, keep_mask.tolist()) if keep
                ]

        # Replace NaN and infinite values with zeros for numerical stability
        clean_data = torch.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Determine sequence length for rolling operations
        seq_len = clean_data.size(1)

        # Dynamische window berekening om te voorkomen dat window > dataset size
        safe_rolling_window = min(rolling_window, seq_len)

        # Create count tensor for expanding-to-rolling window normalization
        counts = torch.arange(1, seq_len + 1, device=clean_data.device).float().unsqueeze(0)
        counts_clamped = torch.clamp(counts, max=safe_rolling_window)

        # Apply normalization logic based on integer flag
        for col_idx, feature in enumerate(current_indices):
            norm_type = active_norms.get(feature.name, 0)
            feature_slice = clean_data[:, :, col_idx]
            
            # TYPE 1: Static Bounded Scale (e.g., / 100 for RSI)
            if norm_type == 1:
                # 50 aftrekken maakt het middelpunt 0. Delen door 50 rekt het uit naar -1 tot +1.
                clean_data[:, :, col_idx] = (feature_slice - 50.0) / 50.0

            # TYPE 2: Dynamic Rolling Z-Score
            elif norm_type == 2:
                # Compute cumulative sums for efficient rolling calculations
                cumsum = torch.cumsum(feature_slice, dim=1)
                cumsum_sq = torch.cumsum(feature_slice ** 2, dim=1)

                # Shift cumulative sums to compute rolling window differences
                shifted_cumsum = torch.zeros_like(cumsum)
                shifted_cumsum[:, safe_rolling_window:] = cumsum[:, :-safe_rolling_window]

                shifted_cumsum_sq = torch.zeros_like(cumsum_sq)
                shifted_cumsum_sq[:, safe_rolling_window:] = cumsum_sq[:, :-safe_rolling_window]

                # Compute rolling window sum and squared sum
                window_sum = cumsum - shifted_cumsum
                window_sum_sq = cumsum_sq - shifted_cumsum_sq

                # Calculate rolling mean and variance
                roll_mean = window_sum / counts_clamped
                roll_var = (window_sum_sq / counts_clamped) - (roll_mean ** 2)

                # Clamp variance to prevent numerical instability
                roll_var = torch.clamp(roll_var, min=1e-8)
                roll_std = torch.sqrt(roll_var)

                # Apply Z-score normalization
                clean_data[:, :, col_idx] = (feature_slice - roll_mean) / roll_std

            # TYPE 3: Rolling MinMax Scaling (Preserves shape, maps strictly to [0, 1])
            elif norm_type == 3:
                unfolded = feature_slice.unfold(1, safe_rolling_window, 1)
                mins = unfolded.min(dim=-1).values
                maxs = unfolded.max(dim=-1).values
                
                # Pad left side to align with sequence length (causal)
                mins = F.pad(mins, (safe_rolling_window - 1, 0), mode='replicate')
                maxs = F.pad(maxs, (safe_rolling_window - 1, 0), mode='replicate')
                
                denom = maxs - mins
                clean_data[:, :, col_idx] = (feature_slice - mins) / torch.clamp(denom, min=1e-8)

            # TYPE 4: Log-Transform (Compresses exponentially growing metrics like Volume)
            elif norm_type == 4:
                clean_data[:, :, col_idx] = torch.sign(feature_slice) * torch.log1p(torch.abs(feature_slice))

            # TYPE 5: Robust Tanh Estimator (Squashes extreme outliers, maps to [-1, 1])
            elif norm_type == 5:
                mu = feature_slice.mean(dim=1, keepdim=True)
                sigma = feature_slice.std(dim=1, keepdim=True)
                clean_data[:, :, col_idx] = torch.tanh(0.5 * ((feature_slice - mu) / torch.clamp(sigma, min=1e-8)))

            elif norm_type == 6:
                # We berekenen wel de spreiding (sigma), maar we trekken het gemiddelde (mu) er NIET af!
                sigma = feature_slice.std(dim=1, keepdim=True)
                clean_data[:, :, col_idx] = torch.tanh(feature_slice / torch.clamp(sigma, min=1e-8))

        # Apply feature-specific clamping after normalization
        for col_idx, feature in enumerate(current_indices):
            clamp_val = active_clamps.get(feature.name, 0.0)

            if clamp_val is not None and clamp_val > 0.0:
                clean_data[:, :, col_idx] = torch.clamp(
                    clean_data[:, :, col_idx],
                    min=-clamp_val,
                    max=clamp_val
                )

        # Output diagnostic statistics if verbose mode is enabled
        if verbose and len(current_indices) > 0:
            logger.info("\n" + "=" * 105)
            logger.info(
                f"{'FEATURE NAME':<30} | {'MEAN':>8} | {'STD':>8} | {'MIN':>8} | {'MAX':>8} | {'DIAGNOSTIC ALERTS'}"
            )
            logger.info("-" * 105)

            # Iterate over each feature column to compute statistics
            for col_idx, feature in enumerate(current_indices):
                col_data = clean_data[:, :, col_idx]

                # Compute statistical metrics
                mean_val = col_data.mean().item()
                std_val = col_data.std(unbiased=False).item() if col_data.numel() > 1 else 0.0
                min_val = col_data.min().item()
                max_val = col_data.max().item()

                alerts: List[str] = []
                norm_type = active_norms.get(feature.name, 0)

                # Flag normalized features with their specific type
                if norm_type == 1:
                    alerts.append("Static Scaled (-50/50) [-1,1]")
                elif norm_type == 2:
                    alerts.append("Rolling Z-Score")
                elif norm_type == 3:
                    alerts.append("Rolling MinMax [0,1]")
                elif norm_type == 4:
                    alerts.append("Log Transform")
                elif norm_type == 5:
                    alerts.append("Robust Tanh (mean)[-1,1]")
                elif norm_type == 6:
                    alerts.append("Robust Tanh (zero-anchored)[-1,1]")

                # Detect statistical outliers
                if (max_val > mean_val + 3 * std_val) or (min_val < mean_val - 3 * std_val):
                    alerts.append("Has Outliers")

                # Suggest normalization if distribution is unstable (only if currently unnormalized)
                if norm_type == 0 and (abs(mean_val) > 0.5 or std_val > 3.0):
                    alerts.append("Needs Norm")

                # Suggest clamping if values exceed reasonable bounds
                if max_val > 10.0 or min_val < -10.0:
                    alerts.append("Needs Clamp")

                # Construct alert string
                alert_str = ", ".join(alerts) if alerts else "HEALTHY"

                # Truncate long feature names for display
                display_name = feature.name[:27] + "..." if len(feature.name) > 30 else feature.name

                # Log formatted diagnostic row
                logger.info(
                    f"{display_name:<30} | {mean_val:>8.3f} | {std_val:>8.3f} | {min_val:>8.3f} | {max_val:>8.3f} | {alert_str}"
                )

            logger.info("=" * 105 + "\n")

        # Return sanitized tensor wrapped in HmoeTensor structure
        return HmoeTensor(
            tensor=clean_data,
            indices=current_indices
        )