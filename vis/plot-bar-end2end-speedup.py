import pandas as pd
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import glob

import os, sys

sys.path.insert(0, os.getcwd())

from plotting import set_size, set_tex_fonts

set_tex_fonts(plt)


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    if row["Compile"] == "jitify":
        return "Jitify"

    if not row["Bounds"]:
        return "DROP"

    if not row["RuntimeConstprop"]:
        return "DROP"

    if row["StoredCache"]:
        return r"Proteus+\$"

    return "Proteus"


def visualize(df, machine, plot_dir, format):
    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = (
        df.groupby(
            [
                "Input",
                "Benchmark",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
            ]
        )
        .mean()
        .reset_index()
    )

    drop_columns = ["Compile", "StoredCache", "Bounds", "RuntimeConstprop", "Input"]

    df["label"] = df.apply(assign_label, axis=1)
    df = df[df.label != "DROP"]
    df = df.drop(columns=drop_columns)

    tmp_df = (
        df.groupby(["Benchmark", "label"]).mean()[["ExeTime", "Speedup"]].reset_index()
    )

    # Detect and fill missing rows (happens for Jitify lulesh)
    missing_df = [
        pd.DataFrame(
            {
                "Benchmark": [benchmark],
                "label": [label],
                "ExecTime": [np.nan],
                "Speedup": [0.0],
            },
        )
        for benchmark, data_df in tmp_df.groupby("Benchmark")
        for label in tmp_df.label.unique()
        if label not in data_df.label.to_list()
    ]
    tmp_df = pd.concat([tmp_df, *missing_df])

    tmp_df = tmp_df.sort_values(by="Benchmark", ascending=True)

    sizes = set_size(width=506.95, fraction=0.5)
    fig, ax = plt.subplots(figsize=sizes)
    bar_width = 0.3
    offset = 0
    bar_order = sorted(tmp_df.label.unique())
    bar_order.remove("AOT")
    spread = bar_width * (len(bar_order) + 1)
    ind = np.arange(0, spread * len(tmp_df.Benchmark.unique()), spread)
    for bar in bar_order:
        # speedup = (
        #    tmp_df[tmp_df.label == "AOT"]["ExeTime"].values
        #    / tmp_df[tmp_df.label == bar]["ExeTime"].values
        # )
        rect = ax.bar(
            ind + offset,
            # speedup,
            tmp_df[tmp_df.label == bar]["Speedup"],
            bar_width,
            label=bar,
        )

        if bar != "AOT":
            bar_labels = ["%.2f" % v for v in tmp_df[tmp_df.label == bar]["Speedup"]]
            # bar_labels = ["%.2f" % v for v in speedup]
            ax.bar_label(
                rect,
                fmt="%g",
                labels=bar_labels,
                padding=0.5,
                fontsize=8,
                rotation=90,
            )

        offset += bar_width

    ax.set_ylabel("Speedup over AOT")
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.set_xticks(ind + bar_width * (len(bar_order) - 1) / 2)
    ax.set_xticklabels(tmp_df.Benchmark.unique())
    yticks = ax.get_yticks()
    ax.set_ylim((yticks.min(), yticks.max()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=15)
    plt.tight_layout()
    ax.legend(
        ncol=len(bar_order),
        bbox_to_anchor=(0, -0.1, 1.0, -0.1),
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=0.5,
        fancybox=False,
        frameon=False,
        shadow=False,
    )
    fn = f"{plot_dir}/bar-end2end-speedup-{machine}.{format}"
    print(f"Storing to {fn}")
    fig.savefig(fn, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot end-to-end speedup")
    parser.add_argument(
        "--dir", help="path to directory containing result files", required=True
    )
    parser.add_argument("--plot-dir", help="directory to store plots in", required=True)
    parser.add_argument(
        "-m",
        "--machine",
        help="which machine to run on: amd|nvidia",
        choices=("amd", "nvidia"),
        required=True,
    )
    parser.add_argument("-f", "--format", help="output image format", default="pdf")
    args = parser.parse_args()

    dfs = list()
    for fn in glob.glob(f"{args.dir}/{args.machine}*-results.csv"):
        df = pd.read_csv(fn, index_col=0)
        found = False
        for sz in ["large", "mid", "small", "default"]:
            if sz in df.Input.unique():
                df = df[df.Input == sz]
                found = True
                break
        assert found, f"In benchmark {df.Benchmark.unique()} we could not deduce input"
        dfs.append(df)

    df = pd.concat(dfs)

    base_durations = df[df["Compile"] == "aot"].set_index(
        ["Benchmark", "Input", "repeat"]
    )["ExeTime"]

    df["Speedup"] = df.apply(
        lambda row: base_durations[row["Benchmark"], row["Input"], row["repeat"]]
        / row["ExeTime"],
        axis=1,
    )
    visualize(df, args.machine, args.plot_dir, args.format)


if __name__ == "__main__":
    main()
