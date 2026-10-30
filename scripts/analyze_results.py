"""Rebuild portfolio metrics and figures from archived evaluations; no training."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "bc": ("BC (single action)", "#64748b"),
    "diffusion_policy_eps": ("Diffusion epsilon (cosine)", "#2563eb"),
    "diffusion_policy_x0": ("Diffusion x0 (cosine)", "#059669"),
    "diffusion_policy_eps_sigmoid": ("Diffusion epsilon (sigmoid)", "#d97706"),
}


def read_run(name):
    path = ROOT / "Q3/results" / name / "eval_metrics.csv"
    with path.open(newline="") as f:
        raw = list(csv.reader(f))
    # Existing training logger may append headers after restart: use last block.
    start = max(i for i, row in enumerate(raw) if row == raw[0])
    header = raw[start]
    rows = [dict(zip(header, row)) for row in raw[start + 1:] if row]
    values = [(int(r["step"]), float(r["mean_success"])) for r in rows]
    if not values or any(not 0 <= score <= 1 for _, score in values):
        raise ValueError(f"Invalid evaluation data: {path}")
    if any(b[0] <= a[0] for a, b in zip(values, values[1:])):
        raise ValueError(f"Evaluation steps must strictly increase: {path}")
    return values


def summarize():
    result = {}
    for name in RUNS:
        rows = read_run(name)
        peak = max(rows, key=lambda row: row[1])
        result[name] = {
            "source": f"Q3/results/{name}/eval_metrics.csv",
            "final_step": rows[-1][0], "final_success": rows[-1][1],
            "best_step": peak[0], "best_success": peak[1],
            "checkpoints": len(rows),
        }
    return result


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    target = ROOT / "docs/assets"
    target.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/results_summary.json").write_text(json.dumps(summarize(), indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    groups = [list(RUNS)[:3], ["diffusion_policy_eps", "diffusion_policy_eps_sigmoid"]]
    for ax, names, title in zip(axes, groups, ["Main comparison", "Bonus: noise schedule"]):
        for name in names:
            label, color = RUNS[name]
            rows = read_run(name)
            ax.plot([s / 1000 for s, _ in rows], [r for _, r in rows],
                    label=label, color=color, linewidth=2, marker="o", markersize=4)
        ax.set(title=title, xlabel="Training updates (thousands)", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Task success rate")
    fig.suptitle("Square-nut assembly | 25 evaluation episodes per checkpoint", fontsize=12)
    fig.tight_layout()
    fig.savefig(target / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
