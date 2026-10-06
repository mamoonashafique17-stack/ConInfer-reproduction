# Reproduction of ConInfer (CVPR 2026 Findings) on LoveDA

Author: Mamoona Shafique

This repository contains my reproduction of **ConInfer: Context-Aware Inference for Training-Free
Open-Vocabulary Remote Sensing Segmentation** (arXiv:2603.29271) on the LoveDA validation set,
run on a free Google Colab T4 GPU.

- Paper: https://arxiv.org/abs/2603.29271
- Original code (not copied here): https://github.com/Dog-Yang/ConInfer
- Report: `report.pdf`

## Results (LoveDA validation, 1669 images)

| Batch size | aAcc | mIoU | mAcc |
|---|---|---|---|
| 8  | 52.86 | 36.99 | 59.82 |
| 25 | 55.43 | 39.02 | 60.87 |
| 50 | 56.15 | **39.78** | 61.38 |

The paper reports 39.33 mIoU (batch/group size 50). Per-class results and discussion are in the report.

## Contents

- `ConInfer_reproduction.ipynb` - Colab notebook (setup, evaluation, batch-size study)
- (the notebook keeps the saved Colab outputs of all runs: batch 8, 25, 50)
- `report.pdf` - the report

## How to reproduce

1. Open `ConInfer_reproduction.ipynb` in Google Colab with a T4 GPU.
2. Run the setup cells in order (clones the original repo, installs dependencies, downloads the
   DINOv3 SAT-493M weights and the LoveDA validation set).
3. Run the batch-size loop cell (batch 8, 25, 50). Each run takes about 4-5 minutes.

All modifications to the original repository are done by the `sed` commands inside the notebook, so no modified source files are stored here.

## What I changed compared with the original setup

- `mmcv-lite` 2.1.0 with a stub for compiled ops instead of the full `mmcv`.
- DINOv3 SAT-493M weights from a third-party Hugging Face repo (`MVRL/dinov3_vitl16_sat`).
- Paths in `ConInfer_segmentor.py` changed to Colab paths; `num_workers` set to 2.
- Only the LoveDA validation split was converted and evaluated.
- `segearth_segmentor.py` was copied from the SegEarth-OV repository.

## Notes

- Results were identical when batch 8 and batch 50 were repeated (deterministic).
- SegEarth-OV and MaskCLIP* baselines in the report are taken from the paper, not re-run.
- Model weights and datasets are not included; see the links above.
