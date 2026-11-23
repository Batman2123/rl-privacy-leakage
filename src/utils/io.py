"""Small file helpers shared across the project."""
import json


def load_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a list of dicts.

    make_dataset.py writes people.jsonl and train_docs.jsonl in this format.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
