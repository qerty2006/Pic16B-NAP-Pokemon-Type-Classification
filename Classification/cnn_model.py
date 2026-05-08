import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


if __name__ == "__main__":
    import torch
    model = build_efficientnet_b0(num_classes=18)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (batch=4, classes=18)")
    print(f"  → apply sigmoid + threshold 0.5 for multi-label predictions")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
