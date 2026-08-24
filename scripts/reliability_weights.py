import numpy as np
import pandas as pd
from pathlib import Path


def load_case_weights(csv_path, weight_mode="e3a", alpha=0.5,
                      class_frequencies=None, epsilon=0.05,
                      num_classes=4):
    """
    Build per-case per-class scalar reliability weights from variance CSV.

    Returns dict: {case_id: np.ndarray of shape (num_classes,)}
    """
    df = pd.read_csv(csv_path)
    case_weights = {}

    for _, row in df.iterrows():
        case_id = row["CaseID"]

        # Extract per-class mean variance (Label_0 = background, Label_1..4 = foreground)
        var = np.array([row[f"Label_{c}"] for c in range(num_classes)],
                       dtype=np.float32)

        # Normalize variance to [0, 1] across classes for this case
        vmin, vmax = var.min(), var.max()
        if vmax - vmin > 1e-8:
            var_norm = (var - vmin) / (vmax - vmin)
        else:
            var_norm = np.zeros_like(var)

        if weight_mode == "e3a":
            # Invert: high variance -> low weight
            weights = 1.0 - var_norm

        elif weight_mode == "e3b":
            assert class_frequencies is not None
            epistemic = 1.0 - var_norm

            # Frequency correction: rare classes get upweighted
            freq_arr = np.ones(num_classes, dtype=np.float32)
            for c, freq in class_frequencies.items():
                freq_arr[c] = freq
            freq_norm = freq_arr / (freq_arr.sum() + 1e-8)
            freq_weight = 1.0 / (freq_norm + 1e-6)
            freq_weight = freq_weight / (freq_weight.max() + 1e-8)

            weights = alpha * epistemic + (1 - alpha) * freq_weight

        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        # Clip so no class is fully silenced
        weights = np.clip(weights, epsilon, 1.0).astype(np.float32)
        case_weights[case_id] = weights

    return case_weights
