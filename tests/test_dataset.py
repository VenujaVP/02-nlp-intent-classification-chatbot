
import json
from pathlib import Path

# Find the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "intents.json"


def test_dataset():

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    intents = data["intents"]

    total_examples = 0

    print("=" * 45)
    print("DATASET VALIDATION")
    print("=" * 45)

    for intent in intents:

        tag = intent["tag"]
        patterns = intent["patterns"]

        count = len(patterns)

        total_examples += count

        print(f"{tag}: {count} examples")

        # Check for duplicate patterns within each intent
        if len(patterns) != len(set(
            pattern.lower().strip() for pattern in patterns
        )):
            print(f"WARNING: Duplicate patterns in {tag}")

    print("-" * 45)

    print("Total intents:", len(intents))
    print("Total examples:", total_examples)

    print("=" * 45)

    if total_examples == 120 and len(intents) == 6:
        print("Dataset validation successful!")
    else:
        print("Dataset validation needs attention!")


if __name__ == "__main__":
    test_dataset()
