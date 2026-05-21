import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Load pretrained EfficientNet-B0 and replace the classifier head for multi-label type prediction.

    The final linear layer is swapped to output num_classes logits (one per type).
    The model outputs raw logits — callers must apply sigmoid before thresholding or top-k selection.

    Args:
        num_classes: Number of output logits. Should be 18 (one per Pokemon type).
        freeze_backbone: If True, only the new classifier head is trained. Use for fast
                         experiments or very small datasets. Default False fine-tunes everything.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace only the final linear layer — all earlier EfficientNet layers keep their ImageNet features
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


if __name__ == "__main__":
    import torch
    model = build_efficientnet_b0(num_classes=18)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (batch=4, classes=18)")
    print(f"  → raw logits; sigmoid applied in train.py/evaluate.py, not inside the model")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
