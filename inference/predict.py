import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import load_model


DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]


def load_image(image_path):
    image = Image.open(image_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image


def main():
    parser = argparse.ArgumentParser(
        description="MIPaaS Chest X-ray disease prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to chest X-ray image"
    )

    parser.add_argument(
        "--model",
        default="densenet121_weighted_aug_cosine",
        choices=[
            "resnet18",
            "densenet121",
            "densenet121_weighted",
            "densenet121_weighted_aug_cosine"
        ],
        help="Model to use"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load thresholds
    threshold_path = os.path.join(
        os.path.dirname(__file__),
        "thresholds.json"
    )

    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)

    if args.model == "densenet121":
        threshold_key = "densenet121"

    elif args.model == "resnet18":
        threshold_key = "densenet121"

    else:
        threshold_key = args.model

    thresholds = threshold_data[threshold_key]

    # Load model
    model = load_model(args.model, device)

    # Load image
    image = load_image(args.image)
    image = image.to(device)

    # Prediction
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)[0]

    probabilities = probabilities.cpu().numpy()

    print("\nMIPaaS Chest X-ray Prediction")
    print("=" * 50)

    predicted_diseases = []

    for disease, probability in zip(DISEASES, probabilities):

        threshold = thresholds[disease]
        predicted = probability >= threshold

        print(
            f"{disease:22s} "
            f"probability={probability:.4f} "
            f"threshold={threshold:.2f} "
            f"prediction={'YES' if predicted else 'NO'}"
        )

        if predicted:
            predicted_diseases.append(disease)

    print("\nPredicted diseases:")
    print("-" * 50)

    if predicted_diseases:
        for disease in predicted_diseases:
            print(f"- {disease}")
    else:
        print("No disease crossed its configured threshold.")


if __name__ == "__main__":
    main()