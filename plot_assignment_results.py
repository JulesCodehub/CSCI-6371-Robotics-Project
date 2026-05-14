#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--decentralized_csv", required=True)
    parser.add_argument("--outdir", default="results/plots")
    parser.add_argument("--combined_csv", default="results/combined_assignment_scores.csv")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined_csv = Path(args.combined_csv)
    combined_csv.parent.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline_csv)
    decentralized = pd.read_csv(args.decentralized_csv)

    baseline["version"] = "Baseline CPFA"
    decentralized["version"] = "Decentralized CPFA"

    df = pd.concat([baseline, decentralized], ignore_index=True)
    df.to_csv(combined_csv, index=False)

    required_columns = {"version", "distribution", "collected_resources"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    distribution_order = ["random", "clustered", "powerlaw"]
    distribution_order = [d for d in distribution_order if d in df["distribution"].unique()]

    data = []
    positions = []
    group_centers = []

    baseline_positions = []
    decentralized_positions = []

    group_gap = 3
    box_offset = 0.35

    for idx, distribution in enumerate(distribution_order):
        center = idx * group_gap + 1
        group_centers.append(center)

        baseline_pos = center - box_offset
        decentralized_pos = center + box_offset

        baseline_positions.append(baseline_pos)
        decentralized_positions.append(decentralized_pos)

        baseline_values = df[
            (df["distribution"] == distribution)
            & (df["version"] == "Baseline CPFA")
        ]["collected_resources"]

        decentralized_values = df[
            (df["distribution"] == distribution)
            & (df["version"] == "Decentralized CPFA")
        ]["collected_resources"]

        data.extend([baseline_values, decentralized_values])
        positions.extend([baseline_pos, decentralized_pos])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    boxplot = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        whiskerprops={"color": "0.25", "linewidth": 1},
        capprops={"color": "0.25", "linewidth": 1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "0.6",
            "markeredgecolor": "0.6",
            "markersize": 3,
            "alpha": 0.45,
        },
    )

    baseline_color = "#BFD7EA"
    decentralized_color = "#C7E9C0"

    for i, patch in enumerate(boxplot["boxes"]):
        if i % 2 == 0:
            patch.set_facecolor(baseline_color)
        else:
            patch.set_facecolor(decentralized_color)

        patch.set_edgecolor("0.25")
        patch.set_linewidth(1)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [d.capitalize() for d in distribution_order],
        fontsize=11
    )

    ax.set_ylabel("Collected Resources", fontsize=11)
    ax.set_xlabel("Food Distribution", fontsize=11)
    ax.set_title(
        "CPFA Performance by Food Distribution",
        fontsize=14,
        fontweight="bold",
        pad=28
    )

    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=baseline_color, edgecolor="0.25", label="Baseline CPFA"),
        Patch(facecolor=decentralized_color, edgecolor="0.25", label="Decentralized CPFA"),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_file = outdir / "grouped_boxplot_all_distributions.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = (
        df.groupby(["distribution", "version"])["collected_resources"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    summary_file = outdir / "summary_statistics.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved combined CSV: {combined_csv}")
    print(f"Saved grouped boxplot: {output_file}")
    print(f"Saved summary statistics: {summary_file}")


if __name__ == "__main__":
    main()