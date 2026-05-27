import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace only the final linear layer — all earlier EfficientNet layers keep their ImageNet features
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


if __name__ == "__main__":
    import torch
    model = build_model(num_classes=18)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (batch=4, classes=18)")
    print(f"  → raw logits; sigmoid applied in train.py/evaluate.py, not inside the model")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
