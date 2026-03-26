# kWTA Sparse ResNet-18 — PyTorch Hub

Pretrained **SparseResNet-18** with **k-Winners-Take-All (kWTA)** activation for adversarial robustness on CIFAR-10.

## Quick Start

```python
import torch

model = torch.hub.load(
    'maurapintor/sparse-resnet18-kwta',
    'sparse_resnet18',
    pretrained=True,
    gamma=0.1,
)
model.eval()

# Run inference on a CIFAR-10 sized input
output = model(torch.randn(1, 3, 32, 32))
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained` | `bool` | `False` | Load adversarially pretrained weights (CIFAR-10, gamma=0.1) |
| `gamma` | `float` | `0.1` | Sparsity ratio applied uniformly to all 4 layers |
| `num_classes` | `int` | `10` | Number of output classes |

## How kWTA Works

The **k-Winners-Take-All (kWTA)** mechanism replaces the standard ReLU activation with a top-k activation function. For each forward pass, only the **k largest activations** per layer are kept (set to their original values); all others are zeroed out. This enforces a fixed sparsity ratio γ across the activations, which has been shown to improve adversarial robustness by making it harder for gradient-based attacks to find adversarial perturbations.

The sparsity ratio γ controls what fraction of activations are preserved:
- `gamma=0.1` → only the top 10% of activations survive per layer

## Pretrained Weights

The pretrained weights are adversarially trained on **CIFAR-10** with `gamma=0.1`.

## Uploading Weights as a GitHub Release

1. Go to [Releases → Create a new release](https://github.com/maurapintor/sparse-resnet18-kwta/releases/new)
2. Set the tag to **`v1.0`**
3. Attach the file **`kwta_spresnet18_0.1_cifar_adv.pth`**
4. Publish the release

The `hubconf.py` entry point will then download weights automatically when `pretrained=True`.
