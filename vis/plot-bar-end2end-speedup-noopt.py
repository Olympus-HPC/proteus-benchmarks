import pandas as pd
from matplotlib.ticker import ScalarFormatter
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import glob
import os
import sys

TEXT_WIDTH = 506.295

sys.path.insert(0, os.getcwd())
from plotting import set_tex_fonts, set_size

set_tex_fonts(plt)


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    if row["Bounds"]:
        return "DROP"

    if row["RuntimeConstprop"]:
        return "DROP"

    if row["StoredCache"]:
        return r"Proteus+\$"

    return "Proteus"


def visualize(df, machine, plot_dir):
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

    bar_order = ["Proteus", r"Proteus+\$"]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color = colors[:8]
    c_map = {
        "AOT": color[0],
        "Proteus": color[1],
        r"Proteus+\$": color[2],
        "Jitify": color[7],
    }

    tmp_df = (
        df.groupby(["Benchmark", "label"]).mean()[["ExeTime", "Speedup"]].reset_index()
    )

    sizes = set_size(TEXT_WIDTH, 0.5)
    fig, ax = plt.subplots(figsize=sizes)
    bar_width = 0.3
    uniq = tmp_df.Benchmark.unique()
    spread = bar_width * (len(bar_order) + 1)
    ind = np.arange(0, spread * len(uniq), spread)[: len(uniq)]
    offset = 0
    for i, bar in enumerate(bar_order):
        rect = ax.bar(
            ind + offset,
            tmp_df[tmp_df.label == bar]["Speedup"],
            bar_width,
            color=c_map[bar],
            label=bar,
        )
        # NOTE: For whatever reason fmt (even when passed with %) does not format
        # to 2 digits. I am rounding now explicitly through pandas functionality
        if bar != "AOT":
            ax.bar_label(
                rect,
                fmt="{:,.2f}",
                labels=tmp_df[tmp_df.label == bar]["Speedup"].round(2).tolist(),
                padding=0.5,
                fontsize=8,
                rotation=90,
            )
        offset += bar_width
    ax.set_ylabel("Speedup over AOT\nw/o JIT opt.")
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
        ncol=2,
        bbox_to_anchor=(0, -0.1, 1.0, -0.1),
        handlelength=1.0,
        handletextpad=0.5,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fn = "{0}/bar-end2end-speedup-noopt-{1}.pdf".format(plot_dir, machine)
    print(f"Storing to {fn}")
    fig.savefig(fn, bbox_inches="tight")
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
    visualize(df, args.machine, args.plot_dir)


if __name__ == "__main__":
    main()
