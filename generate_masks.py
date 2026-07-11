"""
Generate instance segmentation masks for a folder of images.

Inputs a folder of images, and outputs a folder containing .npy numpy
arrays with the masks for each image -- same usage pattern as the
original (Matterport Mask R-CNN), swapped to torchvision's built-in,
actively maintained Mask R-CNN. See DESIGN.md for why.
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

COCO_CATEGORY_NAMES = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
]


def load_model(device):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model


def run_image(model, image_path, device, score_thresh=0.5, mask_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    x = to_tensor(img).to(device)

    with torch.no_grad():
        output = model([x])[0]

    keep = output["scores"] >= score_thresh
    masks = output["masks"][keep, 0] > mask_thresh  # (N, H, W) bool
    labels = output["labels"][keep]
    scores = output["scores"][keep]

    masks_np = masks.cpu().numpy()
    labels_np = [COCO_CATEGORY_NAMES[i] for i in labels.cpu().tolist()]
    scores_np = scores.cpu().numpy()

    return img, masks_np, labels_np, scores_np


def visualize(img, masks_np, labels_np, scores_np):
    overlay = img.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    for i, mask in enumerate(masks_np):
        color = _PALETTE[i % len(_PALETTE)]
        mask_img = Image.fromarray((mask * 100).astype(np.uint8), mode="L")
        colored = Image.new("RGBA", overlay.size, color + (0,))
        colored.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, colored)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            draw.text((xs.min(), ys.min() - 12), f"{labels_np[i]} {scores_np[i]:.2f}", fill=color + (255,))

    overlay = Image.alpha_composite(overlay, draw_layer)
    return overlay.convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--out-dir", default="output_masks")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--save-viz", action="store_true", help="also save a colored overlay .jpg per image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")
    print(f"torchvision {torchvision.__version__}")

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(device)

    image_files = [
        f for f in sorted(os.listdir(args.images_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print(f"no images found in {args.images_dir}")
        return

    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(args.images_dir, fname)

        img, masks_np, labels_np, scores_np = run_image(model, path, device, args.score_thresh)

        mask_path = os.path.join(args.out_dir, f"mask_{stem}.npy")
        np.save(mask_path, masks_np)

        meta_path = os.path.join(args.out_dir, f"mask_{stem}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"labels": labels_np, "scores": scores_np.tolist()}, f, indent=2)

        print(f"{fname}: {len(masks_np)} instances -> {mask_path}")
        for label, score in zip(labels_np, scores_np):
            print(f"    {label:15s} {score:.3f}")

        if args.save_viz:
            viz = visualize(img, masks_np, labels_np, scores_np)
            viz_path = os.path.join(args.out_dir, f"mask_{stem}_viz.jpg")
            viz.save(viz_path)
            print(f"    wrote {viz_path}")


if __name__ == "__main__":
    main()
