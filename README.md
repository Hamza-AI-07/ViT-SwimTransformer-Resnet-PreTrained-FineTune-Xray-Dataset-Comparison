# ViT-SwinTransformer-Resnet-PreTrained-FineTune-Xray-Dataset-Comparison

Professional benchmark project comparing pretrained **ViT**, **Swin Transformer**, and **ResNet152** models after fine-tuning on an X-ray dataset.

## Project Overview

This repository presents a practical comparison of three widely used computer vision model families for medical image classification:

- Vision Transformer (ViT)
- Swin Transformer
- ResNet152

The goal is to evaluate how well each pretrained architecture adapts to X-ray data through transfer learning and fine-tuning, then summarize performance in a reproducible format.

## Objectives

- Build a fair baseline comparison between CNN and Transformer-based models.
- Fine-tune pretrained backbones on the same X-ray task.
- Report final performance in a clear, reproducible manner.
- Provide artifacts and notebook analysis for future extension.

## Repository Structure

- `ViT_Swim_Transformer_Resnet_Comparison.ipynb`: end-to-end experiment workflow and analysis.
- `final_comparison.csv`: final model-wise comparison metrics.
- `model_artifacts/vit/vit_model.pth`: fine-tuned ViT checkpoint.
- `model_artifacts/vit/result.txt`: ViT final result summary.
- `model_artifacts/swin/swin_model.pth`: fine-tuned Swin checkpoint.
- `model_artifacts/swin/result.txt`: Swin final result summary.
- `model_artifacts/resnet/resnet_model.pth`: fine-tuned ResNet checkpoint.
- `model_artifacts/resnet/result.txt`: ResNet final result summary.

## Experimental Setup (Summary)

- Task: X-ray image classification.
- Initialization: pretrained weights (transfer learning).
- Strategy: fine-tuning under a consistent evaluation pipeline.
- Reporting metric in this repository: **Accuracy**.

### Training Configuration

- Input resolution: `224 x 224` (`transforms.Resize((224, 224))`)
- Batch size: `16`
- Epochs: `5`
- Learning rate: `1e-4`
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`

### Model Details

The notebook loads models through `timm.create_model(..., pretrained=True, num_classes=NUM_CLASSES)` using the following architecture IDs:

| Family | Model ID in Notebook | Pretraining Dataset | Key Architecture Details |
|---|---|---|---|
| ViT | `vit_base_patch16_224_in21k` | ImageNet-21k (via timm mapping to `vit_base_patch16_224.augreg_in21k`) | Patch size: `16 x 16`, input size: `224 x 224` |
| Swin Transformer | `swin_base_patch4_window7_224` | ImageNet-1k (timm default pretrained weights) | Patch size: `4 x 4`, window size: `7 x 7`, input size: `224 x 224` |
| ResNet | `resnet152` | ImageNet-1k (timm default pretrained weights) | Backbone: **ResNet-152** (deep CNN with residual bottleneck blocks), classifier head adapted to `NUM_CLASSES` |

In short: ViT starts from **ImageNet-21k** weights, while Swin Transformer and ResNet-152 use **ImageNet-1k** pretrained weights in this setup.

> For preprocessing, training configuration, and evaluation pipeline details, see the notebook.

## Final Results

Extracted from `final_comparison.csv` and per-model `result.txt` files:

| Model | Accuracy |
|------|----------:|
| ViT | 0.7500 |
| Swin Transformer | 0.8221 |
| ResNet | 0.8253 |

## Key Findings

- **Best overall accuracy:** ResNet (`0.8253`).
- **Close second:** Swin Transformer (`0.8221`).
- **Lowest in this setup:** ViT (`0.7500`).
- In this experiment, fine-tuned ResNet slightly outperformed the Transformer-based alternatives on the target X-ray dataset.

## Reproducibility

1. Open `ViT_Swim_Transformer_Resnet_Comparison.ipynb` in VS Code or Jupyter.
2. Install required Python dependencies (for example: PyTorch, torchvision, timm, pandas, numpy, scikit-learn, matplotlib).
3. Run notebook cells to reproduce training/evaluation flow, or use existing checkpoints in `model_artifacts/`.
4. Compare outputs against `final_comparison.csv`.

## Notes

- Large binary checkpoints are typically better managed with Git LFS for long-term collaboration.
- Results may vary with data split, hyperparameters, augmentation policy, and random seed.

## License

This project is licensed under the MIT License. Update the copyright holder name in `LICENSE`.
