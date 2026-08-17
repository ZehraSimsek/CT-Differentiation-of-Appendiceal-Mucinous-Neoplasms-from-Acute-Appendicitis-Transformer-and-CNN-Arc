import torch.nn as nn
from config import PipelineConfig
from .densenet import build_densenet
from .efficientnet import build_efficientnet
from .unet_classifier import build_unet_classifier

def build_model(config: PipelineConfig) -> nn.Module:
    """Factory function to instantiate the 3D model architecture."""
    variant = config.model_name.lower()
    
    if variant == "densenet121":
        model = build_densenet(
            variant=variant,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
        )
    elif variant.startswith("efficientnet"):
        model = build_efficientnet(
            variant=variant,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
        )
    elif variant == "unet_plusplus":
        model = build_unet_classifier(
            variant=variant,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
        )
    else:
        raise ValueError(
            f"Unsupported model '{variant}'. "
            "Supported: 'densenet121', 'efficientnet_b4', 'unet_plusplus'."
        )

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(
        f"  ✓ Built {variant}  |  "
        f"{total_params:,} params  ({trainable_params:,} trainable, {frozen_params:,} frozen)"
    )

    return model
