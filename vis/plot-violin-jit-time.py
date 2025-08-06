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


def visualize(d):
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    # generate some random test data
    print(list(d.values()))
    # plot violin plot

    parts = ax.violinplot(d.values(),
                    showmeans=False,
                    showmedians=True)
    colors=['red','green']
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
    ax.set_title('Proteus Compilation Time for RAJAPerf Benchmarks')
    ax.set_xticks([y + 1 for y in range(len(d))],
                  labels=["AMD HIP", "NVIDIA CUDA"])
    ax.set_ylabel("JIT Compilation Time (ms)")

    fig.savefig("JITViolinPlot", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Print results")
    parser.add_argument(
        "--dir", help="path to directory containing result files", required=True
    )
    parser.add_argument("--plot-dir", help="directory to store plots in", required=True)

    parser.add_argument("-f", "--format", help="output image format", default="pdf")
    parser.add_argument("--plot-title", help="set plot title", default="")
    args = parser.parse_args()
    d = {"amd":[], "nvidia":[]}
    for fn in glob.glob(f"{args.dir}/*-results-profiler.csv"):
      fname =os.path.basename(fn)
      i = 0
      while str(fname )[i] != "-":
          i+=1
      machine = str(fname)[:i]
      df = pd.read_csv(
            fn,
            usecols=[
                "Benchmark",
                "JITCompileTime"
            ],
        )
      d[machine].append(df["JITCompileTime"].iloc[0])

    visualize(d)


if __name__ == "__main__":
    main()
