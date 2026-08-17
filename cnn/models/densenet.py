import torch
import torch.nn as nn
from monai.networks.nets.densenet import DenseNet121

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

class SEDenseNetWrapper(nn.Module):
    """Wrapper to add SE Block before DenseNet's classifier head."""
    def __init__(self, densenet):
        super().__init__()
        self.features = densenet.features
        # DenseNet121 has 1024 output channels in features
        self.se = SEBlock3D(1024)
        self.class_layers = densenet.class_layers
        
    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.class_layers(x)
        return x

def build_densenet(
    variant: str = "densenet121",
    pretrained: bool = False,
    num_classes: int = 2,
    dropout_rate: float = 0.4,
    **kwargs
) -> nn.Module:
    """Instantiate a MONAI 3D DenseNet model with an SE Block.
    
    Using the industry-standard MONAI implementation for full 3D capabilities.
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        dropout_prob=dropout_rate,
    )
    
    # Wrap with SE Block
    model = SEDenseNetWrapper(model)
    
    # Initialize the final layer bias mathematically so it doesn't favor one class initially
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            
    return model
