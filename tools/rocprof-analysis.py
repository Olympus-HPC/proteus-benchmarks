import argparse
import pandas as pd
import cxxfilt
import matplotlib.pyplot as plt

class rocprof:
    def parse(self, fn):
        def get_hash(x):
            try:
                hash_pos = 2
                return cxxfilt.demangle(x.split("$")[hash_pos])
            except IndexError:
                return None

        df = pd.read_csv(fn, sep=",")
        # Rename to match output between rocprof, nvprof.
        df.rename(columns={"KernelName": "Name", "Index": "RunIndex"}, inplace=True)
        df["Duration"] = df["EndNs"] - df["BeginNs"]
        df["Name"] = df["Name"].str.replace(" [clone .kd]", "", regex=False)
        df = df[df.Name.str.contains("fp_")]
        df["Hash"] = df.Name.apply(lambda x: get_hash(x))
        df["Name"] = df.Name.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))
        df.Name = df.Name.str.replace("void lbann::(anonymous namespace)::", "", regex=False)
        df.Name = df.Name.str.extract(r"(.*)\(")
        return df

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess rocprof CSV measurement results"
    )
    parser.add_argument(
        "-f1",
        "--file1",
        default=str,
        help="csv file1",
        required=True,
    )
    parser.add_argument(
        "-f2",
        "--file2",
        default=str,
        help="csv file2",
        required=True,
    )


    args = parser.parse_args()

    parser = rocprof()
    df1 = parser.parse(args.file1)[["Name", "Duration", "Hash"]]
    #print(df1)
    #input("k")
    df2 = parser.parse(args.file2)[["Name", "Duration", "Hash"]]
    print("Sum of df1 times", df1.Duration.sum())
    print("Sum of df2 times", df2.Duration.sum())
    #print(df2)
    #input("k")

    # Assumes we exract the same trace.
    merged_df = df1.copy()
    merged_df.rename(columns={"Duration" : "Duration_1"}, inplace=True)
    merged_df["Duration_2"] = df2.Duration
    #print("merged_df\n", merged_df.to_string())
    #input("k")
    #merged_df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=("_1", "_2"))
    merged_df["Speedup"] = merged_df.Duration_1 / merged_df.Duration_2
    merged_df_sorted = merged_df.sort_values(by="Speedup", ascending=False)
    print("# Best 10\n", merged_df_sorted.head(10).to_string())
    print("# Worst 10\n", merged_df_sorted.tail(10).to_string())
    bin_edges=(0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3)
    ax = merged_df.Speedup.hist(bins=bin_edges, grid=False, rwidth=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    labels = [str(l) for l in bin_edges]
    ax.set_xticks(bin_edges, labels=labels)
    ax.set_xticklabels(labels)

    plt.title("Speedup Distro\n" + args.file1.replace("_8_2048_12288.csv", "") + " VS. " + args.file2.replace("_8_2048_12288.csv", ""))
    plt.xlabel("Speedup")
    plt.ylabel("Frequency")
    plt.savefig("speedup-hist.pdf")

if __name__ == "__main__":
    main()
