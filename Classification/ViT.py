import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit_b16(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load pretrained Vision Transformer (ViT-B/16) and replace the head for multi-label prediction.

    Args:
        num_classes: Number of output logits (e.g., 18 for Pokemon types).
        freeze_backbone: Strongly recommended True initially to use ViT as a stable feature extractor.
    """
    # Using DEFAULT pulls the best available ImageNet weights
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # In ViT, the final classification head is a linear layer stored in model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick sanity check matching your EfficientNet testing format
    model = build_vit_b16(num_classes=18, freeze_backbone=True)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape} (batch=4, types=18)")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")