import torch
import logging
import copy
from typing import List

from hmoe2.tensor import HmoeTensor
from hmoe2.schema import HmoeFeature


# Configure dedicated logger (avoids polluting root logger)
logger = logging.getLogger("HmoeSanitizerAuto")
logger.propagate = False

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class HmoeSanitizerAuto:
    """Automated preprocessing pipeline for HMoE feature tensors.

    This sanitizer performs:
    - Feature filtering based on allowed schema
    - NaN / Inf handling
    - Robust or standard normalization
    - Soft clipping for numerical stability
    - Optional explicit per-feature clamping

    Design goals:
    - Preserve temporal dynamics (no nonlinear squashing like tanh)
    - Stabilize gradients (via clipping)
    - Maintain robustness to outliers (via IQR scaling)

    Methods:
        sanitize: Main preprocessing entrypoint.
    """

    @staticmethod
    def sanitize(
        raw_tensor: HmoeTensor,
        allowed_features: List[HmoeFeature],
        drop_nan_columns: bool = False,
        rolling_window: int = 1080,
        scale_factor: float = 1.0,
        use_robust: bool = True,
        verbose: bool = True
    ) -> HmoeTensor:
        """Sanitize and normalize input tensor.

        Args:
            raw_tensor (HmoeTensor): Input tensor [B, T, F].
            allowed_features (List[HmoeFeature]): Schema defining allowed features.
            drop_nan_columns (bool): Drop columns containing NaNs.
            rolling_window (int): Reserved for future temporal normalization.
            scale_factor (float): Scaling multiplier.
            use_robust (bool): Use robust (median/IQR) scaling if True.
            verbose (bool): Enable logging diagnostics.

        Returns:
            HmoeTensor: Cleaned tensor with updated feature indices.
        """
        # Clone tensor to avoid mutating original data
        clean_data = raw_tensor.tensor.clone()

        # Copy feature metadata (column descriptors)
        current_indices = list(raw_tensor.indices)

        # Tracks which columns to keep
        keep_cols = []

        # Tracks filtered feature metadata
        keep_indices = []

        # Stores per-feature clamp values (if defined)
        active_clamps = {}

        # ---------------------------------------------------------------------
        # 1. FEATURE FILTERING
        # ---------------------------------------------------------------------
        # Keep only features that match allowed schema
        for feature in current_indices:
            col_name = feature.name

            is_allowed = False

            # Match exact or hierarchical feature names (e.g., "price__lag1")
            for allowed_obj in allowed_features:
                if (
                    col_name == allowed_obj.name
                    or col_name.startswith(f"{allowed_obj.name}__")
                ):
                    is_allowed = True

                    # Store clamp config if present
                    if hasattr(allowed_obj, 'clamp') and allowed_obj.clamp is not None:
                        active_clamps[col_name] = allowed_obj.clamp

                    break

            keep_cols.append(is_allowed)

            if is_allowed:
                # Deep copy to avoid modifying original schema
                feat_copy = copy.deepcopy(feature)

                # Propagate clamp setting if applicable
                if col_name in active_clamps and hasattr(feat_copy, 'clamp'):
                    feat_copy.clamp = active_clamps[col_name]

                keep_indices.append(feat_copy)

        # Apply feature mask
        keep_tensor_mask = torch.tensor(
            keep_cols,
            dtype=torch.bool,
            device=clean_data.device
        )

        clean_data = clean_data[:, :, keep_tensor_mask]
        current_indices = keep_indices

        # ---------------------------------------------------------------------
        # 2. NaN / INF HANDLING
        # ---------------------------------------------------------------------
        if drop_nan_columns:
            # Identify columns containing any NaNs
            nan_mask = torch.isnan(clean_data).any(dim=0).any(dim=0)

            if nan_mask.any():
                keep_mask = ~nan_mask

                # Remove problematic columns
                clean_data = clean_data[:, :, keep_mask]

                current_indices = [
                    feat for feat, keep in zip(current_indices, keep_mask.tolist()) if keep
                ]

        # Replace NaN and Inf with safe values
        clean_data = torch.nan_to_num(
            clean_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Tracks normalization strategy per feature
        auto_actions = {}

        # Precompute quantile probabilities for robust scaling
        if use_robust and len(current_indices) > 0:
            q_probs = torch.tensor(
                [0.25, 0.75],
                dtype=clean_data.dtype,
                device=clean_data.device
            )

        # ---------------------------------------------------------------------
        # 3. NORMALIZATION
        # ---------------------------------------------------------------------
        for col_idx, feature in enumerate(current_indices):
            feature_name = feature.name.lower()

            # Extract feature slice [B, T]
            feature_slice = clean_data[:, :, col_idx]

            # Skip label/target columns (must remain unchanged)
            if "target" in feature_name or "label" in feature_name:
                auto_actions[feature.name] = "Bypassed (Target)"
                continue

            # Compute global statistics
            global_mu = feature_slice.mean().item()
            global_sigma = feature_slice.std(unbiased=False).item()

            # Skip near-constant features
            if global_sigma < 1e-6:
                auto_actions[feature.name] = "Bypassed (Flatline)"
                continue

            # Detect whether feature is centered around zero
            is_zero_anchored = abs(global_mu) < (0.15 * global_sigma)

            # ---------------- ROBUST SCALING ----------------
            if use_robust:
                # Compute median per batch element
                median = feature_slice.median(dim=1, keepdim=True).values

                # Compute quartiles
                quantiles = torch.quantile(
                    feature_slice,
                    q_probs,
                    dim=1,
                    keepdim=True
                )

                q25, q75 = quantiles[0], quantiles[1]

                # Interquartile range (robust spread estimate)
                iqr = torch.clamp(q75 - q25, min=1e-8)

                # Convert IQR to std approximation
                robust_sigma = torch.clamp(iqr / 1.35, min=1e-8)

                # Apply scaling (centered or zero-anchored)
                if is_zero_anchored:
                    scaled = feature_slice / (scale_factor * robust_sigma)
                    action = "Robust-Scale (Zero-Anchored)"
                else:
                    scaled = (feature_slice - median) / (scale_factor * robust_sigma)
                    action = "Robust-Scale (Median-Centered)"

            # ---------------- STANDARD SCALING ----------------
            else:
                mu = feature_slice.mean(dim=1, keepdim=True)
                sigma = torch.clamp(
                    feature_slice.std(dim=1, keepdim=True),
                    min=1e-8
                )

                if is_zero_anchored:
                    scaled = feature_slice / (scale_factor * sigma)
                    action = "Std-Scale (Zero-Anchored)"
                else:
                    scaled = (feature_slice - mu) / (scale_factor * sigma)
                    action = "Std-Scale (Mean-Centered)"

            # Apply soft clipping to stabilize gradients
            clean_data[:, :, col_idx] = torch.clamp(
                scaled,
                min=-5.0,
                max=5.0
            )

            auto_actions[feature.name] = action

        # ---------------------------------------------------------------------
        # 4. EXPLICIT FEATURE CLAMPING
        # ---------------------------------------------------------------------
        for col_idx, feature in enumerate(current_indices):
            clamp_val = active_clamps.get(feature.name, 0.0)

            # Apply user-defined clamp if present
            if clamp_val is not None and clamp_val > 0.0:
                clean_data[:, :, col_idx] = torch.clamp(
                    clean_data[:, :, col_idx],
                    min=-clamp_val,
                    max=clamp_val
                )

        # ---------------------------------------------------------------------
        # 5. DIAGNOSTICS / LOGGING
        # ---------------------------------------------------------------------
        if verbose and len(current_indices) > 0:
            logger.info("\n" + "=" * 105)
            logger.info(
                f"{'FEATURE NAME':<30} | {'MEAN':>8} | {'STD':>8} | {'MIN':>8} | {'MAX':>8} | {'AUTO ACTION'}"
            )
            logger.info("-" * 105)

            # Log per-feature statistics
            for col_idx, feature in enumerate(current_indices):
                col_data = clean_data[:, :, col_idx]

                mean_val = col_data.mean().item()
                std_val = (
                    col_data.std(unbiased=False).item()
                    if col_data.numel() > 1 else 0.0
                )
                min_val = col_data.min().item()
                max_val = col_data.max().item()

                action_str = auto_actions.get(feature.name, "None")

                # Truncate long feature names for readability
                display_name = (
                    feature.name[:27] + "..."
                    if len(feature.name) > 30
                    else feature.name
                )

                logger.info(
                    f"{display_name:<30} | "
                    f"{mean_val:>8.3f} | "
                    f"{std_val:>8.3f} | "
                    f"{min_val:>8.3f} | "
                    f"{max_val:>8.3f} | "
                    f"{action_str}"
                )

            logger.info("=" * 105 + "\n")

        # Return cleaned tensor with updated feature metadata
        return HmoeTensor(
            tensor=clean_data,
            indices=current_indices
        )