import torch
import torch.nn as nn
class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D data."""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock3D(out_channels)
        )
    def forward(self, x):
        return self.conv(x)
class UNetClassifier3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout_rate: float = 0.4):
        """
        A custom 3D UNet++ style encoder for classification.
        Mimics UNet dense skip connections by concatenating previous encoder features
        at the same depth, but since it's classification, we just build a dense encoder.
        """
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        out = self.global_pool(x4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
def build_unet_classifier(
    variant: str = "unet_plusplus",
    pretrained: bool = False,
    num_classes: int = 2,
    dropout_rate: float = 0.4,
    **kwargs
) -> nn.Module:
    model = UNetClassifier3D(
        in_channels=1,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return model
