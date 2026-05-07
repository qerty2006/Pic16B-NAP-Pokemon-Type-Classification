import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # replace classifier head for our number of types
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    import torch
    model = build_resnet18(num_classes=18)
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (batch=4, classes=18)")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
