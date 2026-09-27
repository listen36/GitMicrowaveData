"""Create publication-ready figures and an ablation table from formal results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#6E6E6E"


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "legend.fontsize": 8.5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_all(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_figure(values: np.ndarray, summary: dict, output: Path) -> None:
    point = summary["holdout_models"]["logistic_physics"]["test"]["roc_auc"]
    lo, _, hi = summary["bootstrap_auc"]["quantiles_2.5_50_97.5"]
    fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
    ax.hist(values, bins=32, color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.6)
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.12, zorder=0)
    ax.axvline(lo, color=ORANGE, linestyle="--", linewidth=1.3)
    ax.axvline(hi, color=ORANGE, linestyle="--", linewidth=1.3)
    ax.axvline(point, color="black", linewidth=1.8)
    ax.text(point - 0.002, ax.get_ylim()[1] * 0.96, f"Observed AUC = {point:.3f}",
            ha="right", va="top", fontsize=9)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.83,
            f"95% bootstrap CI: {lo:.3f}–{hi:.3f}", ha="center", color=ORANGE)
    ax.set(xlabel="Bootstrap ROC AUC", ylabel="Number of resamples",
           title=f"Held-out test ROC AUC ({len(values):,} stratified bootstrap resamples)")
    ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, output, "Fig_S1_bootstrap_auc")


def specificity_figure(summary: dict, output: Path) -> None:
    values = np.asarray([row["specificity"] for row in summary["groupkfold"]], dtype=float)
    folds = np.arange(1, len(values) + 1)
    mean, sd = values.mean(), values.std(ddof=0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(folds, values, color="#B8B8B8", linewidth=1.2, zorder=1)
    ax.scatter(folds, values, s=62, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    for x, y in zip(folds, values):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8 if y < 0.95 else -15),
                    textcoords="offset points", ha="center", color=BLUE, fontsize=8.5)
    ax.axhline(mean, color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Mean = {mean:.3f}; SD = {sd:.3f}")
    ax.fill_between([0.65, 5.35], max(0, mean - sd), min(1, mean + sd),
                    color=ORANGE, alpha=0.10, label="Mean ± SD")
    ax.set(xticks=folds, xlim=(0.65, 5.35), ylim=(0, 1.06),
           xlabel="GroupKFold fold", ylabel="Specificity",
           title="Fold-specific specificity at development-calibrated thresholds")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, output, "Fig_S2_fold_specificity")


def ablation_table(summary: dict, output: Path) -> None:
    primary = summary["holdout_models"]["logistic_physics"]["test"]
    late = summary["ablations"]["late_time_gate_40_75"]["test"]
    shuffled = summary["ablations"]["angular_order_shuffled"]["test"]
    permutation = summary["ablations"]["within_phantom_label_permutation"]
    performance_rows = [
        ["Original model", primary["roc_auc"], primary["sensitivity"], primary["specificity"], primary["balanced_accuracy"]],
        ["Late-time gate (bins 40–74)", late["roc_auc"], late["sensitivity"], late["specificity"], late["balanced_accuracy"]],
        ["Angular-order shuffle", shuffled["roc_auc"], shuffled["sensitivity"], shuffled["specificity"], shuffled["balanced_accuracy"]],
    ]
    performance_headers = ["Analysis", "ROC AUC", "Sensitivity", "Specificity", "Balanced accuracy"]
    with (output / "Table_S1_ablation_performance.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(performance_headers)
        for name, auc, sens, spec, bal in performance_rows:
            writer.writerow([name, f"{auc:.3f}", f"{sens:.3f}", f"{spec:.3f}", f"{bal:.3f}"])
    performance_cells = [[name, f"{auc:.3f}", f"{sens:.3f}", f"{spec:.3f}", f"{bal:.3f}"]
                         for name, auc, sens, spec, bal in performance_rows]
    fig, ax = plt.subplots(figsize=(8.7, 2.65))
    ax.axis("off")
    ax.set_title("Lesion-information ablation performance", fontsize=12, fontweight="bold", pad=12)
    table = ax.table(cellText=performance_cells, colLabels=performance_headers,
                     cellLoc="center", colLoc="center",
                     colWidths=[0.38, 0.14, 0.15, 0.15, 0.18], bbox=[0, 0.06, 1, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F6F8")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")
    save_all(fig, output, "Table_S1_ablation_performance")

    observed = float(primary["roc_auc"])
    null_mean = float(permutation["mean_auc"])
    permutation_rows = [["Within-phantom label permutation", f"{observed:.3f}",
                         f"{null_mean:.3f}", f"{observed - null_mean:.3f}",
                         str(permutation["n"]), f"{permutation['p_one_sided']:.4f}"]]
    permutation_headers = ["Test", "Observed AUC", "Mean null AUC", "Δ AUC", "Permutations", "One-sided p"]
    with (output / "Table_S2_label_permutation_test.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(permutation_headers)
        writer.writerows(permutation_rows)
    fig, ax = plt.subplots(figsize=(9.2, 2.05))
    ax.axis("off")
    ax.set_title("Within-phantom label-permutation test", fontsize=12, fontweight="bold", pad=12)
    table = ax.table(cellText=permutation_rows, colLabels=permutation_headers,
                     cellLoc="center", colLoc="center",
                     colWidths=[0.34, 0.13, 0.13, 0.10, 0.14, 0.14], bbox=[0, 0.30, 1, 0.50])
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")
    ax.text(0, 0.12, "Null labels were shuffled only within each phantom; Δ AUC = observed AUC − mean null AUC.",
            transform=ax.transAxes, fontsize=8.2, color=GRAY)
    save_all(fig, output, "Table_S2_label_permutation_test")


def time_figure(dataset: Path, summary: dict, output: Path) -> None:
    with np.load(dataset, allow_pickle=False) as z:
        x11, x21, y = np.asarray(z["X11"]), np.asarray(z["X21"]), np.asarray(z["y"], dtype=int)
    info = summary["time_axis"]
    dt = float(info["ifft_spacing_ns"])
    times = np.arange(x11.shape[1]) * dt
    gate_start, gate_stop = float(info["gate_start_ns"]), float(info["gate_stop_edge_ns"])
    last = float(info["last_included_sample_ns"])
    gate = slice(5, 40)
    # Measurements are centred at 0, 5, ..., 355 degrees.  Use half-step
    # cell boundaries so the heatmap cells, ticks, and physical angles agree.
    angle_step = 360.0 / x11.shape[2]
    angle_edges = np.arange(x11.shape[2] + 1) * angle_step - angle_step / 2
    time_edges = times[5:41]
    panels = []
    for array in (x11, x21):
        env = np.abs(np.fft.ifft(array, axis=1))
        env /= np.maximum(env.max(axis=(1, 2), keepdims=True), np.finfo(float).eps)
        no_lesion = env[y == 0].mean(axis=0)[gate]
        lesion = env[y == 1].mean(axis=0)[gate]
        panels.append((no_lesion, lesion, lesion - no_lesion))

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 7.2), constrained_layout=True)
    row_specs = ((panels[0], "S11", "OrRd"), (panels[1], "S21", "Greens"))
    for row, (arrays, port, cmap) in enumerate(row_specs):
        vmax = max(float(arrays[0].max()), float(arrays[1].max()))
        diff_max = max(abs(float(arrays[2].min())), abs(float(arrays[2].max())))
        titles = (f"{port} No lesion", f"{port} Lesion", f"{port} Lesion minus no lesion")
        for col, (array, title) in enumerate(zip(arrays, titles)):
            ax = axes[row, col]
            if col < 2:
                mesh = ax.pcolormesh(angle_edges, time_edges, array, cmap=cmap,
                                     vmin=0, vmax=vmax, shading="flat", rasterized=True)
            else:
                mesh = ax.pcolormesh(angle_edges, time_edges, array, cmap="RdBu_r",
                                     vmin=-diff_max, vmax=diff_max, shading="flat", rasterized=True)
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            if row == 0:
                ax.text(-0.18, 1.05, "abc"[col], transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="bottom")
            ax.set(xlim=(angle_edges[0], angle_edges[-1]), ylim=(gate_start, gate_stop),
                   xticks=(0, 90, 180, 270, 355), xlabel="Antenna angle (degrees)")
            if col == 0:
                ax.set_ylabel("IFFT delay (ns)")
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
    fig.text(0.5, -0.01,
             f"Displayed bins [5,40): {gate_start:.3f} ≤ t < {gate_stop:.3f} ns; "
             f"last included sample = {last:.3f} ns (Δt = {dt:.6f} ns).",
             ha="center", fontsize=8.5, color=GRAY)
    save_all(fig, output, "Fig_S3_time_angle_heatmaps_ns")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.results.read_text(encoding="utf-8"))
    values = np.load(args.bootstrap, allow_pickle=False)
    style()
    bootstrap_figure(values, summary, args.output)
    specificity_figure(summary, args.output)
    ablation_table(summary, args.output)
    time_figure(args.dataset, summary, args.output)
    print(f"Wrote publication figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
