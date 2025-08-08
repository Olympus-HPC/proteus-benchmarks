import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import glob
import itertools
import os
import sys

sys.path.insert(0, os.getcwd())

from plotting import set_size, set_tex_fonts

TEXT_WIDTH = 506.295
set_tex_fonts(plt)


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    if row["StoredCache"]:
        return "DROP"

    if not row["Bounds"] and not row["RuntimeConstprop"] and not row["SpecializeDims"]:
        return r"Proteus/$\emptyset$"

    if row["Bounds"] and row["RuntimeConstprop"] and row["SpecializeDims"]:
        return "Proteus"

    if row["Bounds"] and not row["RuntimeConstprop"] and not row["SpecializeDims"]:
        return "Proteus/LB"

    if row["RuntimeConstprop"] and not row["Bounds"] and not row["SpecializeDims"]:
        return "Proteus/RCF"

    if row["SpecializeDims"] and not row["Bounds"] and not row["RuntimeConstprop"]:
        return "Proteus/DIM"

    return "DROP"


def visualize(df, machine, plot_dir, plot_title, format):
    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df["label"] = df.apply(assign_label, axis=1)
    df = df[df.label != "DROP"]

    df = df[["Benchmark", "label", "Speedup", "SpeedupErr"]]

    bar_order = sorted(df.label.unique())
    bar_order.remove("AOT")

    #tmp_df = df.groupby(["Benchmark", "label"]).mean()[["Speedup"]].reset_index()
    # Detect and fill missing rows (happens for Jitify lulesh)
    missing_df = [
        pd.DataFrame(
            {
                "Benchmark": [benchmark],
                "label": [label],
                "Speedup": [0.0],
                "SpeedupErr": [0.0],
            },
        )
        for benchmark, data_df in df.groupby("Benchmark")
        for label in df.label.unique()
        if label not in data_df.label.to_list()
    ]
    tmp_df = pd.concat([df, *missing_df])

    tmp_df = tmp_df.sort_values(by="Benchmark", ascending=True)

    sizes = set_size(TEXT_WIDTH, 0.5)

    benchmarks = df.Benchmark.unique()
    batch_size = 4
    benchmarks = list(itertools.batched(benchmarks, batch_size))
    for i, batch in enumerate(benchmarks):
        plot_df = tmp_df[tmp_df.Benchmark.isin(batch)]
        fig, ax = plt.subplots(figsize=sizes)
        bar_width = 0.3
        offset = 0
        spread = bar_width * (len(bar_order) + 1)
        ind = np.arange(0, spread * len(batch), spread)[: len(batch)]
        for bar in bar_order:
            rect = ax.bar(
                ind + offset,
                plot_df[plot_df.label == bar]["Speedup"],
                bar_width,
                label=bar,
                zorder=1,
                yerr=plot_df[plot_df.label == bar]["SpeedupErr"],
            )

            if bar != "AOT":
                bar_labels = [
                    "%.2f" % v for v in plot_df[plot_df.label == bar]["Speedup"]
                ]
                ax.bar_label(
                    rect,
                    fmt="%g",
                    labels=bar_labels,
                    padding=0.5,
                    fontsize=8,
                    rotation=90,
                )
            offset += bar_width

        #ax.set_title(plot_title)
        ax.set_ylabel("Speedup over AOT\n(kernel time only)")
        ax.yaxis.set_major_formatter("{x: .1f}")
        ax.set_xticks(ind + bar_width * (len(bar_order) - 1) / 2)
        ax.set_xticklabels(plot_df.Benchmark.unique())
        yticks = ax.get_yticks()
        ax.set_ylim((yticks.min(), yticks.max()))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(rotation=15)
        plt.tight_layout()
        ax.legend(
            ncol=len(bar_order),
            bbox_to_anchor=(-0.2, 1.35),
            loc="upper left",
            handlelength=1.0,
            handletextpad=0.1,
            columnspacing=0.5,
            fontsize=6,
            fancybox=False,
            shadow=False,
            frameon=False,
        )
        fn = f"{plot_dir}/{machine}-bar-kernel-speedup-ablation-{i}.{format}"
        print(f"Storing to {fn}")
        fig.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close(fig)

def add_std_quadrature(s):
    return np.sqrt((s ** 2).sum())

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
    for fn in glob.glob(f"{args.dir}/{args.machine}*-results-profiler.csv"):
        df = pd.read_csv(
            fn,
            usecols=[
                "Name",
                "Duration",
                "Benchmark",
                "Input",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
                "SpecializeDims",
                "RunIndex",
                "repeat",
            ],
        )

        found = False
        df = df[df["Name"].str.contains("RAJA", na=False)]
        df = df.groupby(
            [
                "Benchmark",
                "Name",
                "Input",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
                "SpecializeDims",
                "repeat",
            ])["Duration"].agg(['mean', 'std', 'count'])
        #for sz in ["large", "mid", "small", "default"]:
        #    if sz in df.Input.unique():
        #        df = df[df.Input == sz]
        #        found = True
        #        break
        #assert found, f"In benchmark {df.Benchmark.unique()} we could not deduce input"
        dfs.append(df)

    df = pd.concat(dfs).reset_index()

    # Perform a second aggregation combining the device times of all
    # kernel launches contained within a benchmark
    df = (
        df.groupby(
            [
                "Benchmark",
                "Input",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
                "SpecializeDims",
                "repeat",
            ]
        )[["mean","std","count"]]
        .agg({
            "mean" : "sum",
            # standard deviations must be added in quadrature
            "std" : add_std_quadrature,
            "count" : "min"
            })
        .reset_index()
    )

    df.rename(columns={'mean': 'Duration', 'std' : 'StdDev', "count" : "Count"}, inplace=True)

    # Convert to seconds
    df["Duration"] /= 1e9
    df["StdDev"] /= 1e9

    # Compute speedup
    base_durations = df[df["Compile"] == "aot"].set_index(
        ["Benchmark", "Input", "repeat"]
    )["Duration"]
    base_std_dev = df[df["Compile"] == "aot"].set_index(
        ["Benchmark", "Input", "repeat"]
    )["StdDev"]
    df["Speedup"] = df.apply(
        lambda row: base_durations[row["Benchmark"], row["Input"], row["repeat"]]
        / row["Duration"],
        axis=1,
    )

    df["SpeedupErr"] = df.apply(
        lambda row: (1 / np.sqrt(row["Count"]) ) * row["Speedup"] *
                    np.sqrt((base_std_dev[row["Benchmark"], row["Input"], row["repeat"]] / base_durations[row["Benchmark"], row["Input"], row["repeat"]]) ** 2
                            + ( row["StdDev"] / row["Duration"] ) ** 2),
        axis=1
    )
    visualize(df, args.machine, args.plot_dir, args.plot_title, args.format)


if __name__ == "__main__":
    main()
