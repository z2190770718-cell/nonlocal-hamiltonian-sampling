#!/usr/bin/env python
"""Generate paper-ready mixing-time figures from the analysis outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ALPHAS = [0.1, 0.5, 1.0, 1.9]


def read_summary(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_keys = [key for key in rows[0] if key not in {"eigenvalues_first_five"}]
    return [
        {key: float(row[key]) for key in numeric_keys}
        for row in rows
    ]


def key(alpha: float, prefix: str, epsilon: float) -> str:
    return f"{prefix}_eps_{epsilon:g}"


def select_rows(
    rows: list[dict[str, float]], alphas: list[float]
) -> list[dict[str, float]]:
    by_alpha = {float(row["alpha"]): row for row in rows}
    missing = [alpha for alpha in alphas if alpha not in by_alpha]
    if missing:
        raise ValueError(f"Missing alpha values in summary: {missing}")
    return [by_alpha[alpha] for alpha in alphas]


def plot_lambda(rows: list[dict[str, float]], output_path: Path) -> None:
    alphas = np.asarray([row["alpha"] for row in rows])
    lambda1 = np.asarray([row["lambda1"] for row in rows])

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(alphas, lambda1, "o-", color="tab:blue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\lambda_1(\alpha)$")
    ax.set_title(r"Spectral gap $\lambda_1$")
    ax.grid(True, which="both", alpha=0.25)
    ax.annotate(
        r"$\lambda_1=1$",
        xy=(alphas[0], 1.0),
        xytext=(alphas[0] * 1.05, 1.15),
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_mixing(
    rows: list[dict[str, float]], epsilon: float, output_path: Path
) -> None:
    alphas = np.asarray([row["alpha"] for row in rows])
    t1 = np.asarray([row[key(row["alpha"], "T1_exact", epsilon)] for row in rows])
    t2 = np.asarray(
        [
            row[key(row["alpha"], "T2_exact_zero_momentum", epsilon)]
            for row in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(alphas, t1, "o-", label="First order")
    ax.plot(alphas, t2, "s-", label="Second order")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$T_{\mathrm{mix}}$")
    ax.set_title(rf"Mixing time at $\varepsilon={epsilon:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_relaxation(
    rows: list[dict[str, float]],
    result_dir: Path,
    epsilon: float,
    output_dir: Path,
) -> None:
    for row in rows:
        alpha = float(row["alpha"])
        curve_path = result_dir / f"linearized_curves_alpha_{alpha:g}.npz"
        curves = np.load(curve_path, allow_pickle=True)
        times = curves["time"]
        r1 = curves["R1"]
        r2 = curves["R2_zero_momentum"]

        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        ax.semilogy(times[1:], r1[1:], label="First order")
        ax.semilogy(times[1:], r2[1:], label="Second order")
        ax.axhline(epsilon, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(
            row[key(alpha, "T1_exact", epsilon)],
            color="tab:blue",
            linestyle=":",
            linewidth=1.1,
        )
        ax.axvline(
            row[key(alpha, "T2_exact_zero_momentum", epsilon)],
            color="tab:orange",
            linestyle=":",
            linewidth=1.1,
        )
        ax.set_xscale("log")
        ax.set_xlabel("$t$")
        ax.set_ylabel("Normalized error")
        ax.set_title(rf"$\alpha={alpha:g}$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"paper_relaxation_alpha_{alpha:g}_eps_{epsilon:g}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--epsilon", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_root / "/home/zhengmiaolei/PLOT_FIGURES/Mixing_time_analysis/mixing_time_results"
    rows = read_summary(result_dir / "mixing_time_summary.csv")
    paper_rows = select_rows(rows, PAPER_ALPHAS)

    plot_lambda(
        rows,
        result_dir / "paper_lambda1_vs_alpha.pdf",
    )
    plot_mixing(
        rows,
        args.epsilon,
        result_dir / f"paper_mixing_time_vs_alpha_eps_{args.epsilon:g}.pdf",
    )
    plot_relaxation(
        paper_rows,
        result_dir,
        args.epsilon,
        result_dir,
    )


if __name__ == "__main__":
    main()