import torch
import torch.nn.functional as F
import numpy as np
from torch import autocast, nn
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class AleatoricSegmentationLoss(torch.nn.Module):
    """
    Heteroscedastic aleatoric loss for segmentation.
    Network outputs C logits + 1 log-variance per voxel.
    Loss = CE/(2*sigma^2) + Dice/(2*sigma^2) + 0.5*log(sigma^2)
    """
    def __init__(self, smooth=1e-5, variance_clamp=(-4, 4)):
        super().__init__()
        self.smooth = smooth
        self.variance_clamp = variance_clamp

    def forward(self, logits, logvar, target):
        
        logits = logits.float()
        logvar = logvar.float()
        
        # logits: (B, C, D, H, W), logvar: (B, 1, D, H, W), target: (B, 1, D, H, W)
        logvar = torch.clamp(logvar, min=self.variance_clamp[0], max=self.variance_clamp[1])
        sigma_sq = torch.exp(logvar)  # (B, 1, D, H, W)

        tgt_squeezed = target.squeeze(1).long()  # (B, D, H, W)

        # ---- Cross-entropy (weighted by 1/(2*sigma^2)) ----
        ce_per_voxel = F.cross_entropy(logits, tgt_squeezed, reduction='none')  # (B, D, H, W)
        ce_per_voxel = ce_per_voxel.unsqueeze(1)  # (B, 1, D, H, W)
        loss_ce = (ce_per_voxel / (2 * sigma_sq)).mean()

        # ---- Variance regularisation (prevents infinite variance) ----
        loss_reg = 0.5 * (torch.exp(logvar) - logvar - 1).mean()
        loss_reg = 0.5 * logvar.mean()

        # ---- Soft Dice (also weighted by 1/(2*sigma^2)) ----
        probs = torch.softmax(logits, dim=1)
        tgt_onehot = F.one_hot(tgt_squeezed, num_classes=logits.shape[1]) \
                       .permute(0, 4, 1, 2, 3).float()

        weight_map = 1.0 / (2 * sigma_sq)  # (B, 1, D, H, W)
        dims = (0, 2, 3, 4)

        intersection = (weight_map * probs * tgt_onehot).sum(dims)
        cardinality = (weight_map * (probs + tgt_onehot)).sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_dice = 1.0 - dice_per_class[1:].mean()  # exclude background

        return loss_ce + loss_reg + loss_dice

class AleatoricLossWrapper(torch.nn.Module):
    def __init__(self, base_loss, num_classes):
        super().__init__()
        self.base_loss = base_loss
        self.num_classes = num_classes

    def forward(self, output, target):
        logits = output[:, :self.num_classes]
        logvar = output[:, self.num_classes:]

        return self.base_loss(logits, logvar, target)

class DeepSupervisionAleatoric(torch.nn.Module):
    def __init__(self, loss_fn, weights):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, outputs, targets):
        total = 0.0

        for pred, tgt, w in zip(outputs, targets, self.weights):
            if w == 0:
                continue

            total = total + w * self.loss_fn(pred, tgt)

        return total  
    
class nnUNetTrainerAleatoric(nnUNetTrainerDiceCELoss_noSmooth):
    """
    Trainer that predicts an extra variance channel (aleatoric uncertainty).
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.variance_clamp = (-4, 4)  # clamp to avoid extreme values

    @staticmethod
    def build_network_architecture(plans_manager,
                                   configuration_manager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nnUNetTrainerDiceCELoss_noSmooth.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels + 1,
            enable_deep_supervision)

    def _init_network(self, network):
        # Custom init: set bias of the extra variance channel to -2.0
        # so that sigma^2 = exp(-2) ≈ 0.13 (starting with low uncertainty)
        super()._init_network(network)
        for module in network.modules():
            if isinstance(module, torch.nn.Conv3d) and module.out_channels == self.label_manager.num_segmentation_heads + 1:
                # The last output channel is the variance
                with torch.no_grad():
                    if module.bias is not None:
                        module.bias[-1] = -2.0  # logvar init
                break  # only the final segmentation head

    def _build_loss(self):
        base_loss = AleatoricLossWrapper(
            AleatoricSegmentationLoss(
                variance_clamp=self.variance_clamp
            ),
            self.label_manager.num_segmentation_heads
        )

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()

            weights = np.array(
                [1/(2**i) for i in range(len(scales))]
            )
            weights = weights / weights.sum()

            return DeepSupervisionAleatoric(base_loss, weights)

        return base_loss

    def _compute_loss(self, raw_output, target):
        """
        Helper to compute loss for a single scale.
        raw_output: (B, C+1, ...)
        target:     (B, 1, ...)
        """
        logits = raw_output[:, :self.label_manager.num_segmentation_heads]
        logvar = raw_output[:, self.label_manager.num_segmentation_heads:]
        return self.loss(logits, logvar, target)

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True)
                  for t in batch["target"]]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            outputs = self.network(data)
            loss = self.loss(outputs, target)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.detach().cpu().numpy()}
    
    def validation_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']] \
            if isinstance(batch['target'], list) else batch['target'].to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True):
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        # ---- strip variance channel before online evaluation ----
        if self.enable_deep_supervision:
            output_for_eval = output[0]
            target_for_eval = target[0]
        else:
            output_for_eval = output
            target_for_eval = target

        num_classes = self.label_manager.num_segmentation_heads
        logits_for_eval = output_for_eval[:, :num_classes]

        # standard nnUNet online eval, but using only the segmentation logits
        axes = [0] + list(range(2, logits_for_eval.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(logits_for_eval) > 0.5).long()
        else:
            output_seg = logits_for_eval.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(logits_for_eval.shape, device=logits_for_eval.device,
                                                          dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_eval != self.label_manager.ignore_label).float()
                target_for_eval = torch.clone(target_for_eval)
                target_for_eval[target_for_eval == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target_for_eval[:, -1:]
                target_for_eval = target_for_eval[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_eval, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': loss.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}