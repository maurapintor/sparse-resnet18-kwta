dependencies = ['torch']

import torch
from models.sparse_resnet import SparseResNet18 as _SparseResNet18


def sparse_resnet18(pretrained=False, gamma=0.1, **kwargs):
    """
    SparseResNet-18 with k-Winners-Take-All (kWTA) activation.

    Args:
        pretrained (bool): If True, loads adversarially pretrained weights for CIFAR-10.
        gamma (float): Sparsity ratio applied uniformly to all 4 layers (default: 0.1).
        num_classes (int): Number of output classes (default: 10).

    Returns:
        SparseResNet model instance.
    """
    num_classes = kwargs.get('num_classes', 10)
    model = _SparseResNet18(sparsities=[gamma, gamma, gamma, gamma])

    if pretrained:
        checkpoint_url = (
            'https://github.com/maurapintor/sparse-resnet18-kwta/'
            'releases/download/v1.0/kwta_spresnet18_0.1_cifar_adv.pth'
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)

    if num_classes != 10:
        model.linear = torch.nn.Linear(512, num_classes)

    return model
