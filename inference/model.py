import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 14


def build_densenet121():
    """
    Build the DenseNet-121 architecture used during training.
    Input: 1-channel grayscale image
    Output: 14 disease logits
    """
    model = models.densenet121(weights=None)

    old_conv = model.features.conv0

    model.features.conv0 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        NUM_CLASSES
    )

    return model


def build_resnet18():
    """
    Build the ResNet-18 architecture used during training.
    Input: 1-channel grayscale image
    Output: 14 disease logits
    """
    model = models.resnet18(weights=None)

    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def load_model(model_name, device):
    """
    Load a trained MIPaaS checkpoint.

    Supported models:
        - resnet18
        - densenet121
        - densenet121_weighted
        - densenet121_weighted_aug_cosine
    """

    if model_name == "resnet18":
        model = build_resnet18()
        checkpoint_path = "models/best_resnet18.pth"

    elif model_name == "densenet121":
        model = build_densenet121()
        checkpoint_path = "models/best_densenet121.pth"

    elif model_name == "densenet121_weighted":
        model = build_densenet121()
        checkpoint_path = "models/best_densenet121_weighted.pth"

    elif model_name == "densenet121_weighted_aug_cosine":
        model = build_densenet121()
        checkpoint_path = "models/best_densenet121_weighted_aug_cosine.pth"

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    # These are trusted checkpoints created by our own training runs.
    # weights_only=False is required because the checkpoints contain
    # additional training information such as epoch and optimizer state.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Our training checkpoints contain:
    # {
    #     "epoch": ...,
    #     "model_state_dict": ...,
    #     "optimizer_state_dict": ...,
    #     "val_loss": ...
    # }
    #
    # Also support a raw state_dict just in case.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle checkpoints saved using DataParallel.
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"[OK] {model_name} loaded successfully")
    return model