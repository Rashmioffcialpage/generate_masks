# Design Doc: generate_masks

## Goal

Reproduce the original's task (folder of images in, per-image instance
mask `.npy` files out) with a dependency stack that actually installs and
runs on a machine today.

## Why swap matterport/Mask_RCNN for torchvision

The original is built on
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN), which:

- requires TensorFlow 1.x and an old Keras version, both long past their
  supported lifetime,
- requires manually downloading `mask_rcnn_coco.h5` (a multi-hundred-MB
  weights file) and placing it correctly,
- requires a working `pycocotools` build, which the original README
  itself hedges on ("I should have a version of pycocotools in the
  folder, but feel free to replace it if it doesn't work").

torchvision's `maskrcnn_resnet50_fpn` is the same underlying architecture
(ResNet-50 + FPN backbone, region proposal network, per-region mask head)
trained on the same dataset (COCO), maintained as a first-class part of
PyTorch, and installs with one `pip install torchvision`. Weights download
automatically and are cached (`torch.hub`), no manual step.

## What's the same

- Task: folder of images in, one `.npy` mask array per image out.
- Access pattern: `np.load('output_masks/mask_<name>.npy')`, matching the
  original's documented usage exactly.
- Underlying detector family: Mask R-CNN, trained on COCO, in both cases.

## What's different, concretely

- **Exact mask boundaries will differ from the original's output** on the
  same image — different training runs, slightly different backbone
  weights, and torchvision's specific COCO training recipe don't produce
  bit-identical results to matterport's. Both are legitimate Mask R-CNN
  instances; neither is "the real one."
- **Metadata added.** The original's `.npy` files carry no information
  about what class each mask instance is or how confident the model was.
  `generate_masks.py` here also writes a `_meta.json` with labels +
  scores per image, since a mask array with no idea what it's a mask *of*
  is hard to use for anything beyond "some object was here."
- **Score threshold is a CLI flag** (`--score-thresh`, default 0.5)
  instead of buried in code, since it's the main knob for trading off
  false positives vs. missed detections.

## Known limitations

- **No per-instance NMS across overlapping same-object detections.** The
  test run in the README shows exactly this: one dog produced two
  detections (`dog` 89.3%, `cat` 61.6%) because Mask R-CNN's box-level NMS
  operates per-class, not across classes, so a genuinely ambiguous region
  can produce a kept detection under more than one label. Real usage that
  wants exactly one mask per physical object should either take only the
  highest-scoring detection per heavily-overlapping mask group, or raise
  `--score-thresh` to filter out the weaker duplicate.
- **CPU/MPS inference is much slower than a CUDA GPU** for this model
  (region proposal + per-ROI mask head don't fuse as well on Apple
  Silicon as a simple CNN forward pass does). Fine for a handful of
  images; would need a GPU for a large batch job.
- **No batching across multiple images per forward pass** — each image is
  run through the model individually. torchvision's detection models do
  support batched input; this wasn't implemented since directory-of-images
  workloads are usually I/O-bound on loading anyway at this scale.
