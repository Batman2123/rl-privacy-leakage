"""Turn per-person quiz results into per-group tables and plots."""
import matplotlib

matplotlib.use("Agg")  # save figures to files; works without a display
import matplotlib.pyplot as plt
import pandas as pd

REPEAT_TICKS = [0, 1, 2, 4, 8, 16]


def summarize(forced_df, free_df, k=10):
    """Average each metric within each repeat group (0 = held-out control)."""
    a = forced_df.groupby("repeats").agg(
        n=("name", "size"),
        logprob_local=("logprob_local", "mean"),
        first_top10=("first_token_top10", "mean"),
        all_top10=("all_tokens_top10", "mean"),
        greedy_exact=("greedy_exact", "mean"),
    )
    b = free_df.groupby("repeats").agg(
        recall_at_k=(f"recall@{k}", "mean"),
        refusal_rate=("refusal_rate", "mean"),
    )
    return a.join(b).round(3)


def knows_vs_says(forced_df, free_df, k=10):
    """2x2 table: does the model KNOW the email (forced, all tokens top-10)
    vs. does it SAY it when asked freely (recall@k)? People in the
    knows-but-doesn't-say cell are the ones RL could make leak."""
    m = forced_df.merge(free_df, on=["name", "repeats", "split"])
    return pd.crosstab(m["all_tokens_top10"], m[f"recall@{k}"], margins=True)


def plot_loss(history, path):
    steps, losses = zip(*history)
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("memorization fine-tune")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_before_after(before, after, path):
    """Three panels vs. number of training repeats: forced log-prob, recall
    (forced top-10 and free recall@10), and refusal rate."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for s, lab in [(before, "base"), (after, "after SFT")]:
        ax[0].plot(s.index, s["logprob_local"], "o-", label=lab)
        ax[1].plot(s.index, s["all_top10"], "o-", label=lab + " (forced, top-10)")
        ax[1].plot(s.index, s["recall_at_k"], "s--", label=lab + " (free, recall@10)")
        ax[2].plot(s.index, s["refusal_rate"], "o-", label=lab)
    ax[0].set_title("log-prob of true local part (forced)")
    ax[1].set_title("recall")
    ax[2].set_title("refusal rate (free)")
    for a in ax:
        # symlog so 0 (held-out) and 1..16 fit on one axis
        a.set_xscale("symlog", linthresh=1)
        a.set_xticks(REPEAT_TICKS); a.set_xticklabels(REPEAT_TICKS)
        a.set_xlabel("training repeats"); a.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
