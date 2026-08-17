import torch
import torch.nn as nn

from monai.networks.nets.efficientnet import EfficientNetBN

def build_efficientnet(
    variant: str = "efficientnet_b0",
    pretrained: bool = False,
    num_classes: int = 2,
    dropout_rate: float = 0.4,
    **kwargs
) -> nn.Module:
    """Instantiate a MONAI 3D EfficientNet model.
    
    Using the industry-standard MONAI implementation for full 3D capabilities.
    """
    model = EfficientNetBN(
        model_name=variant.replace("_", "-"),  # monai uses efficientnet-b0
        spatial_dims=3,
        in_channels=1,
        num_classes=num_classes,
    )
    
    # Stabilize the classifier head for cropped/zoomed data
    if hasattr(model, "_fc") and isinstance(model._fc, nn.Linear):
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    # Initialize the final layer bias mathematically so it doesn't favor one class initially
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            
    return model
