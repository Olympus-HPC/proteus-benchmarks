import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import pathlib
import glob

TEXT_WIDTH = 506.295

from plotting import set_tex_fonts, set_size

set_tex_fonts(plt)


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    if row["Compile"] == "jitify":
        return "Jitify"

    if (
        row["StoredCache"]
        and row["Bounds"]
        and row["RuntimeConstprop"]
        and row["SpecializeDims"]
    ):
        return "Proteus"

    return "DROP"


def visualize(df, machine, plot_dir, plot_title, format):
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
                "SpecializeDims",
            ]
        )
        .mean()
        .reset_index()
    )

    df["label"] = df.apply(assign_label, axis=1)
    df = df[df.label != "DROP"]
    df = df[["Benchmark", "label", "Ctime"]]

    # Here we need to get mean/std
    tmp_df = df.groupby(["Benchmark", "label"]).mean()[["Ctime"]].reset_index()

    # Detect and fill missing rows (happens for Jitify lulesh)
    missing_df = [
        pd.DataFrame(
            {
                "Benchmark": [benchmark],
                "label": [label],
                "Ctime": [np.nan],
            },
        )
        for benchmark, data_df in tmp_df.groupby("Benchmark")
        for label in tmp_df.label.unique()
        if label not in data_df.label.to_list()
    ]
    tmp_df = pd.concat([tmp_df, *missing_df])

    tmp_df = tmp_df.sort_values(by="Benchmark", ascending=True)

    sizes = set_size(TEXT_WIDTH, 0.5)
    fig, ax = plt.subplots(figsize=sizes)
    bar_width = 0.3
    offset = 0
    uniq = tmp_df.Benchmark.unique()
    bar_order = sorted(tmp_df.label.unique())
    bar_order.remove("AOT")
    spread = bar_width * (len(bar_order) + 1)
    ind = np.arange(0, spread * len(uniq), spread)[: len(uniq)]
    for bar in bar_order:
        slowdown = (
            tmp_df[tmp_df.label == bar]["Ctime"].values
            / tmp_df[tmp_df.label == "AOT"]["Ctime"].values
        )
        rect = ax.bar(
            ind + offset,
            slowdown,
            bar_width,
            label=bar,
        )

        ax.bar_label(
            rect,
            fmt="{:,.1f}",
            labels=slowdown.round(1),
            padding=0.5,
            fontsize=8,
            rotation=90,
        )
        offset += bar_width

    ax.set_title(plot_title)
    ax.set_ylabel("Slowdown compiling\nAOT+Ext. vs. AOT")
    ax.yaxis.set_major_formatter("{x: .1f}")
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
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    fn = f"{plot_dir}/{machine}-bar-compilation-time-slowdown.{format}"
    print(f"Storing to {fn}")
    fig.savefig(fn, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Print results")
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
    parser.add_argument("--plot-title", help="set plot title", default="")
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
    visualize(df, args.machine, args.plot_dir, args.plot_title, args.format)


if __name__ == "__main__":
    main()
