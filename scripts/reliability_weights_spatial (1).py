import numpy as np
from pathlib import Path

def compute_e3a_weights(case_id, variance_dir, num_classes, epsilon=0.05):
    """
    E3a: Pure epistemic weighting.
    High variance -> low weight. 
    w_v = 1 - variance_normalized, clipped to [epsilon, 1]
    Returns: (num_classes, D, H, W) float32
    """
    path = Path(variance_dir) / f"{case_id}_variance_weights.npz"
    var_map = np.load(path)["variance"]  # (num_classes, D, H, W)
    
    weights = 1.0 - var_map  # invert: uncertain voxels get lower weight
    weights = np.clip(weights, epsilon, 1.0)  # never fully zero
    
    return weights.astype(np.float32)


def compute_e3b_weights(case_id, variance_dir, num_classes, class_frequencies,
                          alpha=0.5, epsilon=0.05):
    """
    E3b: Epistemic weighting blended with inverse-class-frequency weighting.
    Mirrors the scalar e3b formula from reliability_weights.py, applied per-voxel.

    weights = alpha * epistemic + (1 - alpha) * freq_weight

    epistemic:   per-voxel, from variance map (1 - var_norm), same as e3a
    freq_weight: per-class scalar, broadcast over all voxels of that class

    Returns: (num_classes, D, H, W) float32
    """
    path = Path(variance_dir) / f"{case_id}_variance_weights.npz"
    var_map = np.load(path)["variance"]  # (num_classes, D, H, W), already in [0,1]

    epistemic = 1.0 - var_map  # same inversion as e3a, NOT clipped yet

    # --- static per-class frequency weight, identical formula to scalar e3b ---
    freq_arr = np.ones(num_classes, dtype=np.float32)
    for c, freq in class_frequencies.items():
        freq_arr[c] = freq
    # background has no meaningful "rarity" — exclude it from the inverse-frequency calc,
    # rather than letting a placeholder default of 1 distort the normalization
    freq_norm = freq_arr / (freq_arr[1:].sum() + 1e-8)  # normalize over foreground only
    freq_weight = 1.0 / (freq_norm + 1e-6)
    freq_weight[0] = freq_weight[1:].max()  # neutral value, don't let background dominate
    freq_weight = freq_weight / (freq_weight.max() + 1e-8)

    # broadcast freq_weight (num_classes,) over spatial dims to match epistemic
    freq_weight_map = freq_weight[:, None, None, None] * np.ones_like(epistemic)

    weights = alpha * epistemic + (1 - alpha) * freq_weight_map
    weights = np.clip(weights, epsilon, 1.0)  # never fully zero, same as e3a

    return weights.astype(np.float32)