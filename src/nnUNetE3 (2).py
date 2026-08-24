import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from .reliability_weights import load_case_weights


class ReliabilityWeightedLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, tgt, weight_vec):
        """
        pred:       (B, num_classes, D, H, W)
        tgt:        (B, 1, D, H, W)
        weight_vec: (B, num_classes) — per-class scalar weights, 
                    broadcast over spatial dims
        """
        B, C = pred.shape[0], pred.shape[1]

        # Expand weight_vec to full spatial volume: (B, C, 1, 1, 1)
        w = weight_vec.view(B, C, 1, 1, 1)

        # One-hot target: (B, C, D, H, W)
        tgt_squeezed = tgt.squeeze(1).long()
        tgt_onehot = F.one_hot(tgt_squeezed, num_classes=C)\
                       .permute(0, 4, 1, 2, 3).float()

        # Per-voxel weight from ground truth class channel
        voxel_w = (w * tgt_onehot).sum(dim=1)  # (B, D, H, W)

        # Weighted CE
        log_probs = F.log_softmax(pred, dim=1)
        ce = F.nll_loss(log_probs, tgt_squeezed, reduction="none")
        weighted_ce = (ce * voxel_w).mean()

        # Weighted soft Dice
        probs = torch.softmax(pred, dim=1)
        dims = (0, 2, 3, 4)
        intersection = (w * probs * tgt_onehot).sum(dims)
        cardinality  = (w * (probs + tgt_onehot)).sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / \
                         (cardinality + self.smooth)
        weighted_dice = 1.0 - dice_per_class[1:].mean()  # exclude background

        return weighted_ce + weighted_dice


class DeepSupervisionReliability(torch.nn.Module):
    """
    Thin wrapper that passes weight_vec through to each scale.
    """
    def __init__(self, loss_fn, weights):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights  # per-scale DS weights, shape (num_scales,)

    def forward(self, outputs, targets, weight_vec=None):
        total = None
        for i, (pred, tgt, ds_w) in enumerate(
                zip(outputs, targets, self.weights)):
            if ds_w == 0.0:
                continue
            # Use uniform weights during validation if weight_vec not provided
            if weight_vec is None:
                B, C = pred.shape[0], pred.shape[1]
                wv = torch.ones(B, C, device=pred.device)
            else:
                wv = weight_vec
            scale_loss = self.loss_fn(pred, tgt, wv)
            total = ds_w * scale_loss if total is None \
                    else total + ds_w * scale_loss
        return total


class nnUNetTrainerReliabilityWeighted(nnUNetTrainerDiceCELoss_noSmooth):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.variance_csv = Path(
            "/root/notebooks/Challenges/Brats26/PEDs/variance_analysis/label_variance_per_case.csv"
        )
        self.weight_mode = "e3b"   # change to "e3b" for E3b
        self.alpha = 0.5

        # From your E2 TP counts — foreground classes only
        #For PEDs
        self.class_frequencies = {
            1: 4231.806122448979,
            2: 29521.506802721087,
            3: 818.2380952380952,
            4: 2465.187074829932,
        }
        
        #For Glioma
        #self.class_frequencies = {
        #    1: 2051.210810810811,
        #    2: 49943.298841698845,
        #    3: 7196.299613899614,
        #    4: 1663.4200772200772,
        #}

        # Loaded at first train step to avoid init-time IO
        self._case_weights = None

    def _ensure_weights_loaded(self):
        if self._case_weights is None:
            self._case_weights = load_case_weights(
                csv_path=self.variance_csv,
                weight_mode=self.weight_mode,
                alpha=self.alpha,
                class_frequencies=self.class_frequencies,
                num_classes=self.label_manager.num_segmentation_heads
            )
            self.print_to_log_file(
                f"Loaded reliability weights for "
                f"{len(self._case_weights)} cases "
                f"[mode={self.weight_mode}]"
            )

    def _get_weight_vec(self, case_ids):
        """
        Build (B, num_classes) weight tensor for the current batch.
        """
        self._ensure_weights_loaded()
        batch_weights = []
        for cid in case_ids:
            # case_ids from nnU-Net may include full path — take stem
            key = Path(cid).stem
            if key not in self._case_weights:
                # fallback: uniform weights
                n = self.label_manager.num_segmentation_heads + 1
                batch_weights.append(np.ones(n, dtype=np.float32))
                self.print_to_log_file(
                    f"WARNING: no weights for {key}, using uniform")
            else:
                batch_weights.append(self._case_weights[key])
        return torch.from_numpy(np.stack(batch_weights, axis=0))

    def _build_loss(self):
        base_loss = ReliabilityWeightedLoss()
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            ds_weights = np.array(
                [1 / (2 ** i) for i in range(len(scales))])
            ds_weights = ds_weights / ds_weights.sum()
            return DeepSupervisionReliability(base_loss, ds_weights)
        return base_loss

    def train_step(self, batch):
        data   = batch["data"].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True)
                  for t in batch["target"]]
        case_ids = batch["keys"]

        weight_vec = self._get_weight_vec(case_ids).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            loss   = self.loss(output, target, weight_vec)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.detach().cpu().numpy()}