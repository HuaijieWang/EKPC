# EKPC

Code for **EKPC: Elastic Knowledge Preservation and Compensation for Class-Incremental Learning**, International Journal of Computer Vision, 134, article 238, 2026.

[Paper](https://doi.org/10.1007/s11263-026-02820-4) · [arXiv](https://arxiv.org/abs/2506.12351)

EKPC trains shared adapters in a frozen ViT, protects historical capabilities with Importance-aware Parameter Regularization (IPR), and updates a unified classifier with Trainable Semantic Drift Compensation (TSDC). Historical training images are not replayed.

## Installation

Use Python 3.12 and create an isolated environment. The pinned versions below were checked on Linux with CPU PyTorch; they are a repair validation environment, not a recovered record of the original experiments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Select a PyTorch wheel compatible with your GPU driver. For CPU checks, replace the first installation command's index with `https://download.pytorch.org/whl/cpu` and set `"device": [-1]` in the configuration. The paper uses one NVIDIA RTX 3090; this entry point supports one CPU or one CUDA GPU. PyTorch installation options are documented [here](https://pytorch.org/get-started/previous-versions/#v251).

On Windows, activate the environment with `.venv\Scripts\activate`. Start with `"num_workers": 0` if worker startup is problematic.

## Quick start

From the repository root:

```bash
python EKPC/main.py --config EKPC/exps/adapter_cifar224.json
```

The default configuration is CIFAR-100. Data is downloaded automatically through Torchvision. The default backbone is `vit_base_patch16_224_in21k`, obtained through timm with `pretrained=True`.

All data, weight, and output paths in a JSON configuration are resolved relative to the nested `EKPC/` directory. Absolute paths are supported. The configuration file itself is resolved from the invoking directory, with a fallback to `EKPC/`. To use a specific physical GPU, set `CUDA_VISIBLE_DEVICES` and keep `"device": ["0"]`.

For a first full run with one seed:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('EKPC/exps/adapter_cifar224.json')
config = json.loads(path.read_text())
config['seed'] = [1997]
config['prefix'] = 'ekpc_single'
Path('EKPC/exps/cifar_single.json').write_text(json.dumps(config, indent=4) + '\n')
PY
python EKPC/main.py --config EKPC/exps/cifar_single.json
```

Reducing the epoch counts to one is a pipeline check and does not reproduce a paper result.

## Experiments

| Configuration in `EKPC/exps/` | Dataset | Classes | Initial / incremental classes | Tasks |
| --- | --- | --- | --- | --- |
| `adapter_cifar224.json` | CIFAR-100 resized to 224 | 100 | 10 / 10 | 10 |
| `adapter_cub.json` | CUB-200 | 200 | 20 / 20 | 10 |
| `adapter_domainnet.json` | Split-DomainNet | 200 | 20 / 20 | 10 |
| `adapter_imageneta.json` | Split-ImageNet-A | 200 | 20 / 20 | 10 |
| `adapter_imagenetr.json` | Split-ImageNet-R | 200 | 20 / 20 | 10 |

The configurations use SGD, learning rate 0.01, momentum 0.9, weight decay 0.0005, batch size 48, 30 initial epochs, 15 incremental epochs, and 5 classifier calibration epochs. Cosine scale/margin are 20/0.01; `eta1=100`, `eta2=1`, `w1=w2=1`. Adapter bottleneck 64 and seeds `[1997, 1996, 1993]` are retained from the uploaded code. Each configured seed controls Python, NumPy, and PyTorch randomness, including class order; this fixes the former hard-coded PyTorch seed of 1.

For the paper's 20-task ImageNet-A/R setting, copy the relevant configuration and set `init_cls=increment=10`. For its 50/100-task ImageNet-R and DomainNet settings, use 4/2 classes per task. The former ImageNet-A default of 2 classes per task has been changed to the 10-task main setting. For the paper's DINO experiment, set `convnet_type` to `pretrained_vit_b16_224_dino1k_adapter` in the ImageNet-R configuration.

`ipr`, `tsdc`, `ssca`, and `ca` control IPR, trainable drift, prototype compensation, and classifier calibration, respectively. To disable TSDC completely, disable `tsdc` and `ssca`; retain `ca` if testing calibration without compensation. `main.py` accepts `--config`; experiment parameters belong in JSON.

## Data preparation

Set `data_root` to a dataset directory to override the legacy locations. CIFAR-100 uses it as Torchvision's download root. Other datasets use ImageFolder, with the split directories below directly under `data_root`.

| Dataset | Default path relative to `EKPC/` | Splits |
| --- | --- | --- |
| CIFAR-100 | `data/` | Torchvision train/test |
| CUB-200 | `data/datasets/CIL/cub/` | `train`, `test` |
| ImageNet-R | `data/datasets/CIL/imagenet-r/` | `train`, `test` |
| ImageNet-A | `data/datasets/CIL/imagenet-a/imagenet-a/` | `train`, `test` |
| DomainNet | `data/datasets/CIL/domainnet/` | `train`, `val` |

Each split must contain the same class directories, with images inside each class directory. Class mappings are checked before training. The paper selects the 200 DomainNet categories with the most images. Exact original split manifests and DomainNet domain/sampling details were not included in the uploaded repository; a newly chosen split must not be presented as the original split.

Training transforms remain RandomResizedCrop(224), random horizontal flip, and ToTensor; evaluation uses Resize(256), CenterCrop(224), and ToTensor. The uploaded pipeline's lack of Normalize is retained rather than silently changing its input distribution.

## Pretrained weights

Pin timm to the supplied version. Backbone keys are explicitly converted to the existing split Q/K/V attention and MLP layout; missing backbone keys or unexpected keys are rejected. Only adapter parameters are unfrozen.

For offline execution, set `pretrained_path` to a PyTorch state dictionary from the corresponding timm feature extractor with `num_classes=0`. For example, prepare an ImageNet-21K checkpoint in a connected environment:

```python
import timm
import torch

model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
torch.save(model.state_dict(), 'vit_b16_in21k_features.pth')
```

Set an absolute checkpoint path in JSON. This file is a backbone initialization checkpoint, not an EKPC resume checkpoint. Record its SHA-256 alongside experiment results. Weights are not bundled.

## Implementation choices and repair scope

The original folder layout, ViT attention/MLP computation, adapter architecture, cosine classifier, and Gaussian classifier calibration are retained. Changes are restricted to the active EKPC path and removal of unused code.

- IPR statistics are estimated after convergence on current-task training images under evaluation transforms. Local statistics implement token-to-CLS cosine routing, sample means, and task accumulation; global statistics accumulate task means of channel-wise prototype/variance ratios. The up/down matrix scales follow Eq. 19. The old reversed-softmax importance heuristic and batch-dependent loss rescaling are removed. Regularization covers the down/up matrices in Eq. 2.
- The signed accumulated global vector is converted to its magnitude when constructing nonnegative parameter-importance maps. This magnitude convention is explicit: Eq. 6 can be signed for LayerNorm features, whereas Eq. 2 needs nonnegative weights for a preservation penalty. This is a repair interpretation, not a claim that the paper states an absolute-value operator. The released original experiment code is insufficient to resolve this ambiguity exactly.
- Local importance uses the post-training current-task training set once per task, avoiding dependence on epoch count or batch partitioning. No neuron replacement/decay parameters are invented; their unused remnants are removed.
- Compensation uses `new_features - old_features`, followed by addition to old prototypes. This preserves the uploaded code's direction and gives the correct prototype translation. The displayed subtraction in Eq. 20 and addition in Eq. 22 otherwise conflict; this sign choice is documented rather than silently asserting both are identical.
- `drift_sigma=4.0`, Adapter input bias/dropout/scalar, 256 calibration samples per class, and covariance diagonal jitter `1e-3` retain uploaded implementation conventions where the paper does not provide a complete executable specification. The code does not replace the fixed kernel bandwidth with an inferred per-class bandwidth. Optional `samples_per_class` changes the calibration sample count for local checks.
- Old/new compensation features use identical sample ordering and evaluation preprocessing. Drift is differentiable through current features; Gaussian kernel weights use frozen old features. TSD uses the squared drift norm, and the total loss uses the fixed `w1/w2` weights in Eq. 24.
- Classifier calibration samples from compensated prototypes without the former task-dependent prototype shrinkage. Old covariance matrices are retained, as in the uploaded implementation; no unspecified covariance-transport rule is invented.
- Invalid Fisher/LST references, uninitialized first-task losses, device handling, seed handling, and repeated logging setup are fixed. Unused baseline files, VTAB/iBOT/VPT paths, memory/recycling code, dead utilities, comments, and docstrings are removed. Source attribution is retained below.

These are paper-informed repairs to an incomplete release, with the unresolved conventions listed explicitly. They should not be described as a bit-for-bit recovery of the authors' final experiment code.

## Outputs

Each seed writes a separate log and JSON under:

`EKPC/logs/adapter/<dataset>/<log_init_cls>/<increment>/`

`log_init_cls` is 0 when initial and incremental class counts match. Per-run JSON records the class order, top-1/top-5 curves, `A_Last`, and `A_Avg`. A summary JSON in `EKPC/logs/adapter/<dataset>/summary.json` reports all configured runs and their population standard deviations (`ddof=0`). Set `output_dir` to retain experiments in separate directories; repeating the same configuration in the same directory replaces that run's logs/results.

There is no checkpoint-resume or standalone evaluation entry point. The unused checkpoint writer was removed rather than advertising unsupported flags.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests cover pretrained key conversion and freezing, local importance arithmetic and task accumulation, compensation direction and gradients, stable cosine loss, task boundary metrics, two-task training with calibration, and two-seed log/result isolation. They use a small ViT and generated images without downloading datasets or weights.

The optional full-architecture check can be repeated with `EKPC_FULL_VIT_CHECK=1 python -m pytest -q tests`. It also passed on CPU with the full 12-layer ViT-B/16 at 224×224: timm key conversion, initial feature agreement, finite cosine loss, adapter backpropagation, and a frozen backbone. That check used randomly initialized timm weights to exercise the full architecture, not downloaded ImageNet-21K weights. No full benchmark training or paper-accuracy reproduction has been performed.

## Citation and attribution

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

The framework follows SSIAT. The original adapter implementation credits [unify-parameter-efficient-tuning](https://github.com/jxhe/unify-parameter-efficient-tuning), and pretrained ViT implementations/weights are provided through [timm](https://github.com/huggingface/pytorch-image-models). This repair does not introduce a new license or redistribute pretrained weights/datasets.
