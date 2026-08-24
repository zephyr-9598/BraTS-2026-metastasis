import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from .reliability_dataset import nnUNetDatasetWithVariance
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from .reliability_weights_spatial import compute_e3b_weights

class ReliabilityWeightedLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, tgt, weight_map):
        # weight_map: (B, C, d, h, w) — per-class reliability weight, C == num classes incl. background
        
        tgt_squeezed = tgt.squeeze(1).long()  # (B, d, h, w)
        tgt_onehot = F.one_hot(tgt_squeezed, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        #Debugprint
        #for c in range(weight_map.shape[1]):
        #    class_present = (tgt_squeezed == c)
        #    if class_present.any():
        #        w = weight_map[:, c][class_present]
        #        print(f"class {c}: mean={w.float().mean().item():.4f} std={w.float().std().item():.4f} n_voxels={class_present.sum().item()}")
        #    else:
        #        print(f"class {c}: absent this batch")

        # --- CE: gather the target class's weight per voxel ---
        log_probs = F.log_softmax(pred, dim=1)
        ce_loss = F.nll_loss(log_probs, tgt_squeezed, reduction="none")  # (B, d, h, w)
        ce_voxel_weights = torch.gather(
            weight_map, dim=1, index=tgt_squeezed.unsqueeze(1)
        ).squeeze(1)  # (B, d, h, w) — weight of each voxel's own target class
        weighted_ce = (ce_loss * ce_voxel_weights).mean()

        # --- Dice: per-class weight applied directly, per-class ---
        dims = (0, 2, 3, 4)
        probs = torch.softmax(pred, dim=1)
        intersection = (weight_map * probs * tgt_onehot).sum(dims)
        cardinality = (weight_map * (probs + tgt_onehot)).sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        weighted_dice = 1.0 - dice_per_class[1:].mean()  # exclude background, as before

        return weighted_ce + weighted_dice


class nnUNetTrainerReliabilityWeighted_spatial(nnUNetTrainerDiceCELoss_noSmooth):

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        # 1. Define attributes before parent hooks run
        self.variance_dir = Path("/root/notebooks/Challenges/Brats26/PEDs/variance_analysis/variance_maps2")
        
        # 2. Run the parent initialization
        super().__init__(plans, configuration, fold, dataset_json, device)
    
    def _build_network_architecture(self, architecture_class_name,
                                      arch_init_kwargs, arch_init_kwargs_req_import,
                                      num_input_channels, enable_deep_supervision):
        return super()._build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            4,  # image channels only, weight channels stripped in train_step
            enable_deep_supervision
        )

    def _build_loss(self):
        print("BUILD LOSS CALLED", flush=True)
        base_loss = ReliabilityWeightedLoss()

        if self.enable_deep_supervision:
            # 1. Get the raw downsampling scales (This is what exists!)
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # 2. Compute exponential decay weights like native nnU-Net v2
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            
            # 3. Guard against unused parameter crashes if you use DDP (Distributed Data Parallel)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # 4. Normalize weights to sum up to 1.0
            weights = weights / weights.sum()
            
            # 5. Pack your custom loss inside the wrapper using the computed float weights
            self.loss = DeepSupervisionWrapper(base_loss, weights)
        else:
            self.loss = base_loss
        return self.loss
        
        print("self.loss is now:", self.loss, flush=True)

    def get_tr_and_val_datasets(self):
        # mirror the base implementation but swap in the variance-aware dataset
        tr_keys, val_keys = self.do_split()
        dataset_tr = nnUNetDatasetWithVariance(
            self.preprocessed_dataset_folder, tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            variance_dir=self.variance_dir
        )
        dataset_val = nnUNetDatasetWithVariance(
            self.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            variance_dir=self.variance_dir
        )
        return dataset_tr, dataset_val

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch["target"]]

        img_data   = data[:, :4]                              # (B, 4, D, H, W)
        weight_map = data[:, 4:].clamp(min=0.05, max=1.0)    # (B, 5, D, H, W)

        weight_maps_scaled = []
        for t in target:
            spatial_size = t.shape[2:]
            if weight_map.shape[2:] != spatial_size:
                scaled = F.interpolate(weight_map, size=spatial_size,
                                       mode='trilinear', align_corners=False)
            else:
                scaled = weight_map
            weight_maps_scaled.append(scaled)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(img_data)
            loss = self.loss(output, target, weight_maps_scaled)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return {"loss": loss.detach().cpu().numpy()}
    
    def perform_actual_validation(self, save_probabilities=False):
        # Swap to 4-channel backup folder for validation inference
        original_folder = self.preprocessed_dataset_folder

        # Point to backup which has clean 4-channel data
        self.preprocessed_dataset_folder = str(
            Path(original_folder).parent / 
            "nnUNetPlans_3d_fullres_backup"
        )

        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            # Always restore, even if validation crashes
            self.preprocessed_dataset_folder = original_folder

    
    def validation_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch["target"]]

        num_image_channels = 4

        img_data   = data[:, :num_image_channels]
        weight_map = data[:, num_image_channels:].clamp(min=0.05, max=1.0)

        weight_maps_scaled = []
        for t in target:
            spatial_size = t.shape[2:]
            if weight_map.shape[2:] != spatial_size:
                scaled = F.interpolate(weight_map, size=spatial_size,
                                       mode='trilinear', align_corners=False)
            else:
                scaled = weight_map
            weight_maps_scaled.append(scaled)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(img_data)
            del img_data
            l = self.loss(output, target, weight_maps_scaled)

        # rest of validation metric computation unchanged from your existing code
        if self.enable_deep_supervision:
            output = output[0]
            target_0 = target[0]

        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, 
                                                      device=output.device, 
                                                      dtype=torch.float16)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        mask = None
        if self.label_manager.has_ignore_label:
            mask = (target_0 != self.label_manager.ignore_label).float()
            target_0[target_0 == self.label_manager.ignore_label] = 0

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, 
                                          target_0, axes=axes, mask=mask)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy()[1:],
            'fp_hard': fp.detach().cpu().numpy()[1:],
            'fn_hard': fn.detach().cpu().numpy()[1:]
        }

class nnUNetTrainerSpatialE3b(nnUNetTrainerReliabilityWeighted_spatial):
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, device: str = 'cuda'):
        
        # 1. Intercept and alter the plans identifier before initializing
        plans['plans_name'] = 'nnUNetPlans_3b'
        
        # 2. Call the parent initialization with the modified plans dict
        super().__init__(plans, configuration, fold, dataset_json, device)


#class nnUNetTrainerspatialFreqWeighted_AdultGlioma(nnUNetTrainerspatialFreqWeighted):
#    class_frequencies = {
#        1: 2051.210810810811,
#        2: 49943.298841698845,
#        3: 7196.299613899614,
#        4: 1663.4200772200772,
#    }
