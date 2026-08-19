"""
utils/xai.py — Explainable AI (Grad-CAM 3D)
===========================================
Generates 3D Class Activation Maps using PyTorch hooks to 
visually prove where the model is looking.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class GradCAM3D:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = self._find_target_layer(model)
        self.activations = None
        self.gradients = None
        self.handlers = []
        if self.target_layer is not None:
            self.handlers.append(self.target_layer.register_forward_hook(self.save_activation))
            self.handlers.append(self.target_layer.register_full_backward_hook(self.save_gradient))
        else:
            print("  [XAI WARNING] Could not find a suitable layer for Grad-CAM.")
    def _find_target_layer(self, model: nn.Module) -> nn.Module | None:
        """Find the best Conv3d layer. We prefer 8x8x8 or 16x16x16 spatial sizes for better GradCAM resolution."""
        for name, module in model.named_modules():
            if name == "enc4.conv.3": return module               
            if name == "features.transition3.conv": return module 
            if name == "_blocks.4.10._project_conv": return module 
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                target_layer = module
        return target_layer
    def save_activation(self, module, input, output):
        self.activations = output
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray | None:
        """
        Generate 3D Grad-CAM heatmap for a given input tensor and class.
        input_tensor: [B, C, D, H, W]
        Returns: [D, H, W] numpy array normalized to [0, 1]
        """
        if self.target_layer is None:
            return None
        self.model.eval()
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        self.model.zero_grad()
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            return None
        b, k, d, h, w = self.gradients.size()
        alpha = self.gradients.view(b, k, -1).mean(dim=2).view(b, k, 1, 1, 1)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True) 
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
        input_size = input_tensor.shape[2:] 
        cam = F.interpolate(cam, size=input_size, mode='trilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam
    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
def overlay_heatmap_on_slice(
    original_volume: np.ndarray, 
    heatmap_volume: np.ndarray, 
    save_path: str,
    slice_idx: int | None = None
) -> None:
    """
    Finds the CT depth slice with the maximum activation (or uses slice_idx),
    applies a 'jet' colormap, and saves the overlaid image using matplotlib.
    original_volume: [D, H, W] in [0, 1]
    heatmap_volume: [D, H, W] in [0, 1]
    """
    if slice_idx is None:
        slice_sums = np.sum(heatmap_volume, axis=(1, 2))
        slice_idx = int(np.argmax(slice_sums))
    orig_slice = original_volume[slice_idx]
    heatmap_slice = heatmap_volume[slice_idx]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(orig_slice, cmap='gray', interpolation='nearest')
    plt.imshow(heatmap_slice, cmap='jet', alpha=0.4, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
