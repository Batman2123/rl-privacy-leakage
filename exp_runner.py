"""
Main experiment runner -- iteration 1 (memorization baseline).

Steps:
  1. Build the fake employee dataset (data/make_dataset.py) if it's missing.
  2. Load Qwen3 (non-thinking) and quiz it BEFORE fine-tuning (baseline).
  3. LoRA fine-tune it on the fake directory documents.
  4. Quiz it again AFTER fine-tuning.
  5. Save per-person CSVs, per-group summaries, the knows-vs-says table,
     the loss curve, the before/after plot, and the LoRA adapter.

Usage (from the repo root):
  python exp_runner.py                         # full run, Qwen3-1.7B
  python exp_runner.py --model Qwen/Qwen3-4B-Instruct-2507
  python exp_runner.py --limit 20 --k 2        # quick smoke test

Next step (iteration 2, separate script): GRPO on FACTS-Parametric starting
from the saved adapter, then rerun the same quizzes.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import (forced_probe, free_probe, knows_vs_says,
                         plot_before_after, plot_loss, summarize)
from src.privleak import add_lora, finetune_memorize
from src.utils import load_jsonl


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results/iter1")
    ap.add_argument("--seed", type=int, default=0, help="dataset seed")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--k", type=int, default=10, help="samples per person in the free quiz")
    ap.add_argument("--limit", type=int, default=None,
                    help="only quiz N random people (for quick tests)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:  # record settings for reproducibility
        json.dump(vars(args), f, indent=2)

    # 1. Dataset (deterministic given --seed)
    data = Path(args.data_dir)
    if not (data / "people.jsonl").exists():
        subprocess.run([sys.executable, str(data / "make_dataset.py"),
                        "--seed", str(args.seed), "--out", str(data)], check=True)
    people = load_jsonl(data / "people.jsonl")
    docs = load_jsonl(data / "train_docs.jsonl")
    # --limit: quiz a random subset (spread across all repeat groups) for quick tests
    quiz_people = random.Random(0).sample(people, args.limit) if args.limit else people

    # 2. Model. bf16 where supported, fp16 on older GPUs (e.g. Colab T4).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)

    def quiz(tag):
        forced = forced_probe(model, tok, quiz_people)
        free = free_probe(model, tok, quiz_people, k=args.k)
        forced.to_csv(out / f"forced_{tag}.csv", index=False)
        free.to_csv(out / f"free_{tag}.csv", index=False)
        summary = summarize(forced, free, k=args.k)
        summary.to_csv(out / f"summary_{tag}.csv")
        print(f"\n=== {tag} ===\n{summary}\n")
        return forced, free, summary

    print("Quizzing base model...")
    _, _, s_base = quiz("base")

    # 3. Memorization fine-tune
    model = add_lora(model, r=args.lora_r, alpha=2 * args.lora_r)
    history = finetune_memorize(model, tok, docs, epochs=args.epochs, lr=args.lr,
                                batch_size=args.batch_size)
    plot_loss(history, out / "loss.png")
    model.save_pretrained(out / "sft_adapter")

    # 4. Quiz again
    print("Quizzing fine-tuned model...")
    forced_sft, free_sft, s_sft = quiz("sft")

    # 5. Plots + knows-vs-says table
    plot_before_after(s_base, s_sft, out / "before_after.png")
    kvs = knows_vs_says(forced_sft, free_sft, k=args.k)
    kvs.to_csv(out / "knows_vs_says_sft.csv")
    print(f"Knows (forced top-10) vs. says (free recall@{args.k}):\n{kvs}")
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
