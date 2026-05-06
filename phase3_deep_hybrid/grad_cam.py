#!/usr/bin/env python3
"""Grad-CAM for visualizing CNN attention on pressure maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transfer_learning import TransferLearningModel
from cnn_model import PressureDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "features"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        for name, module in model.named_modules():
            if name.endswith(target_layer):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self.model.eval()
        output = self.model(x)
        self.model.zero_grad()
        target = output[:, target_class]
        target.backward(torch.ones_like(target))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap


def visualize_gradcam(model_path: str | Path, csv_path: str | Path, output_path: str | Path | None = None):
    """Generate and save a Grad-CAM visualization."""

    model = TransferLearningModel().to(DEVICE)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = PressureDataset(Path(csv_path).parent, augment=False)
    if len(dataset) == 0:
        print("no samples in dataset")
        return None

    sample_image, sample_label = dataset[0]
    sample_image = sample_image.unsqueeze(0).to(DEVICE)

    grad_cam = GradCAM(model, target_layer="features")
    heatmap = grad_cam(sample_image, target_class=1)

    heatmap_resized = np.kron(heatmap, np.ones((8, 8)))

    if output_path is None:
        output_path = FIGURE_DIR / "gradcam_visualization.png"
    output_path = Path(output_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(sample_image.squeeze().cpu().numpy(), cmap="gray")
    axes[0].set_title("original pressure map")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="hot")
    axes[1].set_title("grad-cam heatmap")
    axes[1].axis("off")

    axes[2].imshow(sample_image.squeeze().cpu().numpy(), cmap="gray", alpha=0.6)
    axes[2].imshow(heatmap_resized, cmap="hot", alpha=0.4)
    axes[2].set_title("overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print("saved grad-cam visualization to", output_path)
    plt.close(fig)

    return heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    visualize_gradcam(args.model_path, args.csv_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
