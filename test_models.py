import torch
from inference.model import (
    build_densenet121,
    build_resnet18,
)


def test_model(model, checkpoint_path):
    print(f"\nTesting: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # Our training checkpoints contain:
    # epoch, model_state_dict, optimizer_state_dict, val_loss
    state_dict = checkpoint["model_state_dict"]

    # Handle DataParallel checkpoints if necessary
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()

    print("✓ Checkpoint loaded successfully")


test_model(
    build_densenet121(),
    "models/best_densenet121.pth"
)

test_model(
    build_densenet121(),
    "models/best_densenet121_weighted.pth"
)

test_model(
    build_densenet121(),
    "models/best_densenet121_weighted_aug_cosine.pth"
)

test_model(
    build_resnet18(),
    "models/best_resnet18.pth"
)

print("\n🎉 ALL FOUR MODELS LOADED SUCCESSFULLY!")