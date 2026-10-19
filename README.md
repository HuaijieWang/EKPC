# EKPC

**EKPC: Elastic Knowledge Preservation and Compensation for Class-Incremental Learning**
International Journal of Computer Vision, 134, article 238, 2026.

[Paper](https://doi.org/10.1007/s11263-026-02820-4) · [arXiv](https://arxiv.org/abs/2506.12351)

EKPC combines Importance-aware Parameter Regularization (IPR) and Trainable Semantic Drift Compensation (TSDC) for class-incremental learning with shared adapters and a frozen ViT backbone.

## Installation

Python 3.12. Dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Choose a [PyTorch wheel](https://pytorch.org/get-started/previous-versions/#v251) compatible with your GPU driver. For CPU execution, use the `https://download.pytorch.org/whl/cpu` index and set `"device": [-1]`. On Windows, activate the environment with `.venv\Scripts\activate`.

## Training

From the repository root:

```bash
python EKPC/main.py --config EKPC/exps/adapter_cifar224.json
```

CIFAR-100 and the default ImageNet-21K pretrained ViT-B/16 weights are downloaded automatically. Configure the dataset, device, and seeds in the JSON file. For a single run, set `"seed": [1997]`. To select a physical GPU, set `CUDA_VISIBLE_DEVICES` and use `"device": ["0"]`.

Data, weight, and output paths in JSON are relative to `EKPC/`; absolute paths are also supported. Configuration paths are resolved from the invoking directory, with a fallback to `EKPC/`.

| Configuration in `EKPC/exps/` | Dataset | Classes | Classes per task | Tasks |
| --- | --- | --- | --- | --- |
| `adapter_cifar224.json` | CIFAR-100 | 100 | 10 | 10 |
| `adapter_cub.json` | CUB-200 | 200 | 20 | 10 |
| `adapter_domainnet.json` | Split-DomainNet | 200 | 20 | 10 |
| `adapter_imageneta.json` | Split-ImageNet-A | 200 | 20 | 10 |
| `adapter_imagenetr.json` | Split-ImageNet-R | 200 | 20 | 10 |

Default settings: SGD, learning rate 0.01, momentum 0.9, weight decay 0.0005, batch size 48, 30 initial epochs, 15 incremental epochs, and 5 classifier calibration epochs. Adapter bottleneck: 64; cosine scale/margin: 20/0.01; `eta1=100`, `eta2=1`, `w1=w2=1`. Seeds: `[1997, 1996, 1993]`.

For 20-task ImageNet-A/R experiments, set `init_cls=increment=10`. For 50/100-task ImageNet-R and DomainNet experiments, use 4/2 classes per task. For DINO pretraining, set `convnet_type` to `pretrained_vit_b16_224_dino1k_adapter`.

The `ipr`, `tsdc`, `ssca`, and `ca` switches control IPR, trainable drift, prototype compensation, and classifier calibration. Disable both `tsdc` and `ssca` to disable TSDC completely.

## Data preparation

Set `data_root` in JSON to specify a dataset directory, or use the defaults below.

| Dataset | Default directory relative to `EKPC/` | Splits |
| --- | --- | --- |
| CIFAR-100 | `data/` | Torchvision train/test |
| CUB-200 | `data/datasets/CIL/cub/` | `train`, `test` |
| ImageNet-R | `data/datasets/CIL/imagenet-r/` | `train`, `test` |
| ImageNet-A | `data/datasets/CIL/imagenet-a/imagenet-a/` | `train`, `test` |
| DomainNet | `data/datasets/CIL/domainnet/` | `train`, `val` |

Except for CIFAR-100, prepare each split in ImageFolder format: `<data_root>/<split>/<class>/<image>`. All splits must share the same class directories. DomainNet uses the 200 categories with the most images. Original split manifests and detailed DomainNet sampling information are not bundled.

## Pretrained weights

The default backbone is timm's `vit_base_patch16_224_in21k`. For offline execution, set `pretrained_path` to the corresponding timm feature-extractor state dictionary with `num_classes=0`:

```python
import timm
import torch

model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
torch.save(model.state_dict(), 'vit_b16_in21k_features.pth')
```

## Outputs

Per-seed logs and JSON metrics are written to `EKPC/logs/adapter/<dataset>/<log_init_cls>/<increment>/`, where `log_init_cls=0` when initial and incremental class counts match. JSON files contain the class order, top-1/top-5 curves, `A_Last`, and `A_Avg`.

`EKPC/logs/adapter/<dataset>/summary.json` contains the multi-seed mean and standard deviation (`ddof=0`). Set `output_dir` to separate experiments; repeated runs in the same location replace matching outputs.

<details>
<summary>Reproducibility notes</summary>

- Training uses RandomResizedCrop(224), horizontal flipping, and ToTensor; evaluation uses Resize(256), CenterCrop(224), and ToTensor. Normalize is not applied.
- Importance statistics are collected once per task after training. Parameter-importance maps use the magnitude of the accumulated global vector; this nonnegative-weight convention is additional to the signed expression in Eq. 6.
- Prototype compensation adds the weighted `new_features - old_features` displacement, with identical sample ordering and evaluation transforms.
- Kernel bandwidth is `drift_sigma=4.0`; calibration uses 256 samples per class and covariance diagonal jitter `1e-3`. Old covariance matrices are retained.
- The supplied environment has been checked on CPU. Full benchmark reproduction is not yet verified. Checkpoint resume and standalone evaluation are not provided.

</details>

## Citation

```bibtex
@article{wang2026ekpc,
  title={EKPC: Elastic Knowledge Preservation and Compensation for Class-Incremental Learning},
  author={Wang, Huaijie and Cheng, De and He, Lingfeng and Li, Yan and Li, Jie and Wang, Nannan and Gao, Xinbo},
  journal={International Journal of Computer Vision},
  volume={134},
  number={5},
  pages={238},
  year={2026},
  doi={10.1007/s11263-026-02820-4}
}
```

## Acknowledgments

This project builds on SSIAT, [unify-parameter-efficient-tuning](https://github.com/jxhe/unify-parameter-efficient-tuning), and [timm](https://github.com/huggingface/pytorch-image-models).
