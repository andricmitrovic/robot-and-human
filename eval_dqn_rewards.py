# compute_optimal_gap_metrics.py
import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List

# ---------- Configuration ----------
OP_TYPES: List[str] = ["avg", "noisy", "improving"]

DATA_DIR = "./output"
OUT_DIR = "./output/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

_EPS = 1e-12  # avoid divide-by-zero


def _paired_ci(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    2-sided CI on the mean via normal approximation (OK for n >= ~30).
    Returns (low, high).
    """
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    m = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    if np.isnan(se):
        return np.nan, np.nan
    return m - z * se, m + z * se


def compute_metrics(dqn_times: np.ndarray, opt_times: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics when arrays contain POSITIVE total execution times (lower is better).
    Definitions:
      - gap: Δ = DQN - OPT  (Δ > 0 => DQN slower/worse)
      - efficiency ratio: η = OPT / DQN  (η ≤ 1, closer to 1 is better)
    """
    assert dqn_times.shape == opt_times.shape, "DQN and optimal arrays must have the same shape."
    n = dqn_times.size

    # Gap
    gap = dqn_times - opt_times
    mean_gap = float(np.mean(gap))
    std_gap = float(np.std(gap, ddof=1))
    gap_ci_lo, gap_ci_hi = _paired_ci(gap)

    # Efficiency
    safe_dqn = np.where(np.abs(dqn_times) < _EPS, np.sign(dqn_times) * _EPS, dqn_times)
    efficiency = opt_times / safe_dqn
    mean_eff = float(np.mean(efficiency))
    std_eff = float(np.std(efficiency, ddof=1))
    eff_ci_lo, eff_ci_hi = _paired_ci(efficiency)

    # Distributions
    dqn_mean, dqn_std = float(np.mean(dqn_times)), float(np.std(dqn_times, ddof=1))
    opt_mean, opt_std = float(np.mean(opt_times)), float(np.std(opt_times, ddof=1))

    # Probability of exact optimality
    prob_opt = float(np.mean(dqn_times == opt_times))

    # Worst/best gap
    worst_gap = float(np.max(gap))
    best_gap = float(np.min(gap))

    return {
        "n": n,
        "dqn_mean": dqn_mean,
        "dqn_std": dqn_std,
        "opt_mean": opt_mean,
        "opt_std": opt_std,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "gap_ci_low": gap_ci_lo,
        "gap_ci_high": gap_ci_hi,
        "mean_efficiency": mean_eff,
        "std_efficiency": std_eff,
        "efficiency_ci_low": eff_ci_lo,
        "efficiency_ci_high": eff_ci_hi,
        "prob_optimal": prob_opt,
        "worst_gap": worst_gap,
        "best_gap": best_gap,
    }


def pretty_print_report(op_type: str, m: Dict[str, float]) -> None:
    print("=" * 72)
    print(f"Operator type: {op_type}")
    print("-" * 72)
    print(f"Samples (episodes):                  {m['n']}")
    print("")
    print("Execution-time distributions (lower = better):")
    print(f"  DQN   mean ± std:                   {m['dqn_mean']:.6f} ± {m['dqn_std']:.6f}")
    print(f"  Opt   mean ± std:                   {m['opt_mean']:.6f} ± {m['opt_std']:.6f}")
    print("")
    print("Gap to optimal:  Δ = DQN - OPT   (Δ > 0 ⇒ DQN slower/worse)")
    print(f"  mean ± std:                         {m['mean_gap']:.6f} ± {m['std_gap']:.6f}")
    print(f"  95% CI on mean:                     [{m['gap_ci_low']:.6f}, {m['gap_ci_high']:.6f}]")
    print(f"  best (most negative gap):           {m['best_gap']:.6f}")
    print(f"  worst (largest positive gap):       {m['worst_gap']:.6f}")
    print("")
    print("Normalized efficiency:  η = OPT / DQN   (closer to 1.0 is better)")
    print(f"  mean ± std:                         {m['mean_efficiency']:.6f} ± {m['std_efficiency']:.6f}")
    print(f"  95% CI on mean:                     [{m['efficiency_ci_low']:.6f}, {m['efficiency_ci_high']:.6f}]")
    print("")
    print(f"Probability of exact optimality:      {m['prob_optimal']:.6f}")
    print("=" * 72)
    print("")


def save_csv(op_type: str, m: Dict[str, float]) -> str:
    df = pd.DataFrame([m])
    path = os.path.join(OUT_DIR, f"metrics_{op_type}.csv")
    df.to_csv(path, index=False)
    return path


def latex_row(op_type: str, m: Dict[str, float]) -> str:
    """
    Build a LaTeX table row summarizing key numbers per operator type.
    """
    return (
        f"{op_type} & "
        f"{m['dqn_mean']:.3f} ± {m['dqn_std']:.3f} & "
        f"{m['opt_mean']:.3f} ± {m['opt_std']:.3f} & "
        f"{m['mean_gap']:.3f} [{m['gap_ci_low']:.3f},{m['gap_ci_high']:.3f}] & "
        f"{m['mean_efficiency']:.3f} [{m['efficiency_ci_low']:.3f},{m['efficiency_ci_high']:.3f}] & "
        f"{m['prob_optimal']:.3f} \\\\"
    )


def run_one(op_type: str) -> Dict[str, float]:
    dqn_path = os.path.join(DATA_DIR, f"dqn_rewards_{op_type}.npy")
    opt_path = os.path.join(DATA_DIR, f"optimal_rewards_{op_type}.npy")
    if not os.path.exists(dqn_path):
        raise FileNotFoundError(f"Missing file: {dqn_path}")
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Missing file: {opt_path}")

    dqn = np.load(dqn_path)
    opt = np.load(opt_path)

    m = compute_metrics(dqn, opt)
    pretty_print_report(op_type, m)
    csv_path = save_csv(op_type, m)
    print(f"Saved per-operator CSV → {csv_path}")
    return m


if __name__ == "__main__":
    rows = []
    any_written = False
    for op in OP_TYPES:
        try:
            m = run_one(op)
            rows.append(latex_row(op, m))
            any_written = True
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    if any_written:
        latex_table = r"""
\begin{table}[ht]
\centering
\small
\begin{tabular}{lccccc}
\toprule
Operator & DQN (mean ± std) & Opt (mean ± std) & Gap mean [95\% CI] & Eff. mean [95\% CI] & P(match) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Comparison of DQN total execution time to per-episode oracle (lower is better). Gap is $\Delta = \text{DQN} - \text{OPT}$; efficiency is $\eta = \text{OPT}/\text{DQN}$ (closer to 1 is better).}
\label{tab:dqn_vs_optimal_times}
\end{table}
"""
        tex_path = os.path.join(OUT_DIR, "metrics_table.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_table.strip() + "\n")
        print(f"LaTeX table saved → {tex_path}")
    else:
        print("No rows written (no operator files found).")
