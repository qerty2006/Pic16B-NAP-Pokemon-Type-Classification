import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace only the final linear layer — all earlier EfficientNet layers keep their ImageNet features
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class ScratchCNN(nn.Module):
    """
    A custom Convolutional Neural Network built from scratch.
    Designed for 3-channel input images (e.g., 224x224).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 3 x H x W
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/2 x W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/4 x W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/8 x W/8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/16 x W/16
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_scratch_model(num_classes: int) -> nn.Module:
    """
    Builds and returns the custom CNN model built from scratch.
    """
    return ScratchCNN(num_classes)


if __name__ == "__main__":
    print("Testing EfficientNet model:")
    model_eff = build_model(num_classes=18)
    x = torch.randn(4, 3, 224, 224)
    out_eff = model_eff(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out_eff.shape}  (batch=4, classes=18)")
    total_eff = sum(p.numel() for p in model_eff.parameters())
    trainable_eff = sum(p.numel() for p in model_eff.parameters() if p.requires_grad)
    print(f"Params: {total_eff:,} total, {trainable_eff:,} trainable\n")

    print("Testing Custom Scratch CNN model:")
    model_scratch = build_scratch_model(num_classes=18)
    out_scratch = model_scratch(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out_scratch.shape}  (batch=4, classes=18)")
    total_scratch = sum(p.numel() for p in model_scratch.parameters())
    trainable_scratch = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    print(f"Params: {total_scratch:,} total, {trainable_scratch:,} trainable")

