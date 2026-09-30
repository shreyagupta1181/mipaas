import subprocess
import sys


IMAGES = [
    "normal1.jpeg",
    "normal2.jpeg",
    "disease.jpeg",
]

MODELS = [
    "densenet121",
    "densenet121_weighted",
    "densenet121_weighted_aug_cosine",
    "resnet18",
]


def main():
    total = len(IMAGES) * len(MODELS)
    current = 0

    print("=" * 70)
    print("MIPaaS - FULL INFERENCE TEST")
    print(f"Testing {len(IMAGES)} images × {len(MODELS)} models = {total} runs")
    print("=" * 70)

    for image in IMAGES:
        print("\n")
        print("#" * 70)
        print(f"IMAGE: {image}")
        print("#" * 70)

        for model in MODELS:
            current += 1

            print("\n")
            print("-" * 70)
            print(f"RUN {current}/{total}")
            print(f"Image : {image}")
            print(f"Model : {model}")
            print("-" * 70)

            result = subprocess.run(
                [
                    sys.executable,
                    "inference/predict.py",
                    "--image",
                    image,
                    "--model",
                    model,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(result.stdout)
            else:
                print("❌ INFERENCE FAILED")
                print(result.stdout)
                print(result.stderr)

    print("\n")
    print("=" * 70)
    print("FULL INFERENCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()