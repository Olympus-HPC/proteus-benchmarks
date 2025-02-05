import argparse
import pandas as pd
from pathlib import Path
import pathlib
import subprocess
import os
import cxxfilt
import time
import pprint
import re
from itertools import product
import shutil
import json
import tomllib


class ProteusConfig:
    def check_valid(self, key, values):
        if key not in self.valid_keys:
            raise Exception(f"Invalid key {key} not in {self.valid_keys}")

        if not all([o in ["0", "1"] for o in values]):
            raise Exception(f"Expected values 0 or 1 for opt {key}, values: {values}")

    def __init__(self, **kwargs):
        self.valid_keys = [
            "ENV_PROTEUS_USE_STORED_CACHE",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS",
            "ENV_PROTEUS_SPECIALIZE_ARGS",
            "ENV_PROTEUS_SPECIALIZE_DIMS",
        ]
        # Check expected
        for key, values in kwargs.items():
            self.check_valid(key, values)
        # Check all valid keys are present.
        if list(kwargs.keys()) != self.valid_keys:
            raise Exception(
                f"Expected all keys {self.valid_keys} are defined but found only: {list(kwargs.keys())}"
            )
        # Generate all combinations of values
        keys = kwargs.keys()
        values = kwargs.values()
        combinations = product(*values)

        # Create a list of dictionaries from the combinations
        self.env_configs = [
            dict(zip(keys, combination)) for combination in combinations
        ]

    def get_env_configs(self):
        return self.env_configs


class AOTConfig:
    def get_env_configs(self):
        return [
            {
                "ENV_PROTEUS_USE_STORED_CACHE": "0",
                "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
                "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
                "ENV_PROTEUS_SPECIALIZE_DIMS": "0",
            }
        ]


class JitifyConfig:
    def get_env_configs(self):
        return [
            {
                "ENV_PROTEUS_USE_STORED_CACHE": "0",
                "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
                "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
                "ENV_PROTEUS_SPECIALIZE_DIMS": "0",
            }
        ]


class Rocprof:
    def __init__(self, metrics, cwd):
        self.metrics = metrics
        if metrics:
            metrics_file = f"{cwd}/vis-scripts/rocprof-metrics.txt"
            self.command = f"rocprof -i {metrics_file}" + " --timestamp on -o {0} {1}"
        else:
            self.command = "rocprof --timestamp on -o {0} {1}"

    def get_command(self, output, executable):
        return self.command.format(output, executable)

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
        df["Hash"] = df.Name.apply(lambda x: get_hash(x))
        df["Name"] = df.Name.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))
        return df


class Nvprof:
    def __init__(self, metrics):
        if metrics:
            self.command = "nvprof --metrics inst_per_warp,stall_exec_dependency --print-gpu-trace --normalized-time-unit ns --csv --log-file {0} {1}"
        else:
            self.command = "nvprof --print-gpu-trace --normalized-time-unit ns --csv --log-file {0} {1}"
        self.metrics = metrics

    def get_command(self, output, executable):
        return self.command.format(output, executable)

    def parse(self, fn):
        def get_hash(x):
            try:
                hash_pos = 2
                return cxxfilt.demangle(x.split("$")[hash_pos])
            except IndexError:
                return None

        # Skip the first 3 (or 4 lines if metrics are collected) of nvprof
        # metadata info.
        skiprows = 4 if self.metrics else 3
        df = pd.read_csv(fn, sep=",", skiprows=skiprows)
        # Skip the first row after the header which contains units of metrics.
        df = df[1:]
        # Nvprof with metrics tracks only kernels.
        if self.metrics:
            df["Kernel"] = df.Kernel.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))
            df.rename(columns={"Kernel": "Name"}, inplace=True)
        else:
            df["Hash"] = df.Name.apply(lambda x: get_hash(x))
            df["Name"] = df.Name.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))

        return df


class Executor:
    def __init__(self, benchmark, path, executable_name, extra_args, exemode,
                build_command, inputs, cc, proteus_path, env_configs):
        self.benchmark = benchmark
        self.path = path
        self.executable_name = executable_name
        self.extra_args = extra_args
        self.exemode = exemode
        # the build command is meant to be a full bash command to build the benchmark, eg
        # `cmake -DCMAKE_BUILD_TYPE=Debug --build` or `make benchmark`
        # If none is provided, it will default to `make`
        self.build_command = 'make' if build_command == None else build_command
        self.inputs = inputs
        self.cc = cc
        self.proteus_path = proteus_path
        self.env_configs = env_configs

    def __str__(self):
        return f"{self.benchmark} {self.path} {self.exemode}"

    def execute_command(self, cmd, **kwargs):
        print("=> Execute", cmd)
        try:
            p = subprocess.run(
                cmd, check=True, text=True, capture_output=True, shell=True, **kwargs
            )
        except subprocess.CalledProcessError as e:
            print("Failed cmd", e.cmd)
            print("ret", e.returncode)
            print("stdout\n", e.stdout)
            print("stderr\n", e.stderr)
            print(e)
            raise e

        print("=========== stdout ===========")
        print(p.stdout)
        print("==============================")
        print("=========== stderr ===========")
        print(p.stderr)
        print("==============================")
        return p.stdout, p.stderr

    def clean(self):
        os.chdir(self.path)
        cmd = "make clean"
        self.execute_command(cmd)

    def build(self, do_jit):
        os.chdir(self.path)
        env = os.environ.copy()
        env["ENABLE_PROTEUS"] = "yes" if do_jit else "no"
        env["PROTEUS_PATH"] = self.proteus_path
        env["CC"] = self.cc
        t1 = time.perf_counter()
        print(
            "Build command",
            self.build_command,
            "CC=" + env["CC"],
            "PROTEUS_PATH=" + env["PROTEUS_PATH"],
            "ENABLE_PROTEUS=" + env["ENABLE_PROTEUS"],
        )
        if not isinstance(self.build_command, list):
            self.build_command = [self.build_command]
        for cmd in self.build_command:
            self.execute_command(cmd, env=env)
        t2 = time.perf_counter()
        return t2 - t1

    def build_and_run(self, reps, profiler=None):
        os.chdir(self.path)

        results = pd.DataFrame()
        caching = pd.DataFrame()
        assert (
            self.exemode == "aot"
            or self.exemode == "proteus"
            or self.exemode == "jitify"
        ), "Expected aot or proteus or jitify for exemode"

        #self.clean()
        print("BUILD", self.path, "type", self.exemode)
        ctime = self.build(self.exemode != "aot")
        exe_size = Path(f"{self.path}/{self.executable_name}").stat().st_size
        print("=> BUILT")

        for repeat in range(0, reps):
            for input_id, args in self.inputs.items():
                for env in self.env_configs:
                    cmd_env = os.environ.copy()
                    for k, v in env.items():
                        cmd_env[k] = v
                    cmd = f"{self.executable_name} {args} {self.extra_args}"

                    set_launch_bounds = (
                        False if env["ENV_PROTEUS_SET_LAUNCH_BOUNDS"] == "0" else True
                    )
                    use_stored_cache = (
                        False if env["ENV_PROTEUS_USE_STORED_CACHE"] == "0" else True
                    )
                    specialize_args = (
                        False if env["ENV_PROTEUS_SPECIALIZE_ARGS"] == "0" else True
                    )

                    specialize_dims = (
                        False if env["ENV_PROTEUS_SPECIALIZE_DIMS"] == "0" else True
                    )

                    if self.exemode == "proteus":
                        print("Proteus env", env)

                    # Delete any previous generated Proteus stored cache.
                    if use_stored_cache:
                        # Delete amy previous cache files in the command path.
                        shutil.rmtree(".proteus")
                        # Execute a warmup run if using the stored cache to
                        # generate the cache files.  CAUTION: We need to create
                        # the cache jit binaries right before running.
                        # Especially, Proteus launch bounds, runtime args,
                        # specialized dims will be baked into the binary so we
                        # need a "warmup" run for each setting before taking the
                        # measurement.
                        self.execute_command(
                            cmd,
                            env=cmd_env,
                            cwd=str(self.path),
                        )

                    stats = f"{os.getcwd()}/{self.exemode}-{input_id}-{time.time()}.csv"
                    if profiler:
                        # Execute with profiler on.
                        cmd = profiler.get_command(stats, cmd)

                    t1 = time.perf_counter()
                    out, _ = self.execute_command(
                        cmd,
                        env=cmd_env,
                        cwd=str(self.path),
                    )
                    t2 = time.perf_counter()

                    # Cleanup from a stored cache run, removing cache files.
                    cache_size_obj = 0
                    cache_size_bc = 0
                    if use_stored_cache:
                        for file in Path(self.path).glob(".proteus/cache-jit-*.o"):
                            # Size in bytes.
                            cache_size_obj += file.stat().st_size
                        for file in Path(self.path).glob(".proteus/cache-jit-*.bc"):
                            # Size in bytes.
                            cache_size_bc += file.stat().st_size
                        # Delete amy previous cache files in the command path.
                        shutil.rmtree(".proteus")

                    if profiler:
                        df = profiler.parse(stats)
                        os.remove(stats)
                        # Add new columns to the existing dataframe from the
                        # profiler.
                        df["Benchmark"] = self.benchmark
                        df["Input"] = input_id
                        df["Compile"] = self.exemode
                        df["Ctime"] = ctime
                        df["StoredCache"] = use_stored_cache
                        df["Bounds"] = set_launch_bounds
                        df["RuntimeConstprop"] = specialize_args
                        df["SpecializeDims"] = specialize_dims
                        df["ExeSize"] = exe_size
                        df["ExeTime"] = t2 - t1
                        # Drop memcpy operations (because Proteus adds DtoH copies
                        # to read kernel bitcodes that interfere with unique
                        # indexing and add RunIndex for nvprof to uniquely
                        # identify kernel invocations.
                        if isinstance(profiler, Nvprof):
                            df.drop(
                                df[df.Name.str.contains("CUDA memcpy")].index,
                                inplace=True,
                            )
                            # Reset index to sequential, integer index.
                            df.reset_index(drop=True, inplace=True)
                            df["RunIndex"] = df.index
                    else:
                        # Create a new dataframe row.
                        df = pd.DataFrame(
                            {
                                "Benchmark": [self.benchmark],
                                "Input": [input_id],
                                "Compile": [self.exemode],
                                "Ctime": [ctime],
                                "StoredCache": [use_stored_cache],
                                "Bounds": [set_launch_bounds],
                                "RuntimeConstprop": [specialize_args],
                                "SpecializeDims": [specialize_dims],
                                "ExeSize": [exe_size],
                                "ExeTime": [t2 - t1],
                            }
                        )
                    df["repeat"] = repeat
                    results = pd.concat((results, df), ignore_index=True)

                    # Skip parsing caching stats when running AOT.
                    if self.exemode != "proteus":
                        continue

                    # Parse Proteus caching info.
                    matches = re.findall(
                        "HashValue ([0-9]+) NumExecs ([0-9]+) NumHits ([0-9]+)",
                        out,
                    )
                    cache_df = pd.DataFrame(
                        {
                            "HashValue": [str(m[0]) for m in matches],
                            "NumExecs": [int(m[1]) for m in matches],
                            "NumHits": [int(m[2]) for m in matches],
                        }
                    )
                    cache_df["Benchmark"] = self.benchmark
                    cache_df["Input"] = input_id
                    cache_df["StoredCache"] = use_stored_cache
                    cache_df["Bounds"] = set_launch_bounds
                    cache_df["RuntimeConstprop"] = specialize_args
                    cache_df["SpecializeDims"] = specialize_dims
                    cache_df["repeat"] = repeat
                    cache_df["CacheSizeObj"] = cache_size_obj
                    cache_df["CacheSizeBC"] = cache_size_bc

                    caching = pd.concat((caching, cache_df))

        return results, caching


def main():
    parser = argparse.ArgumentParser(
        description="Build, run and collect measurements for a benchmark program"
    )
    parser.add_argument(
        "-t",
        "--toml",
        default=str,
        help="input toml descriptors for benchmarks",
        required=True,
    )
    parser.add_argument(
        "-c", "--compiler", help="path to the compiler executable", required=True
    )
    parser.add_argument(
        "-j",
        "--proteus-path",
        help="path to proteus install directory",
        required=True,
    )
    parser.add_argument(
        "-x",
        "--exemode",
        help="execution mode",
        choices=("aot", "proteus", "jitify"),
        required=True,
    )
    parser.add_argument(
        "-p",
        "--profmode",
        help="profiling mode",
        choices=("direct", "profiler", "metrics"),
        required=True,
    )
    parser.add_argument(
        "-m",
        "--machine",
        help="the machine running on: amd|nvidia",
        choices=("amd", "nvidia"),
        required=True,
    )
    parser.add_argument(
        "-r",
        "--reps",
        help="number of repeats per experiment",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-l",
        "--list",
        help="list available benchmarks and configurations",
        action="store_true",
    )
    parser.add_argument(
        "-b", "--bench", help="run a particular benchmark", nargs="+", default=[]
    )
    parser.add_argument(
        "--proteus-config",
        help="proteus env var configuration",
        type=json.loads,
    )
    parser.add_argument(
        "--suffix", help="add custom suffix to save CSV files", default=""
    )
    args = parser.parse_args()

    with open(args.toml, "rb") as f:
        benchmark_configs = tomllib.load(f)

    if args.list:
        pprint.pprint(benchmark_configs)
        return

    for bench in args.bench:
        if bench not in benchmark_configs.keys():
            raise Exception(
                f"{bench} not in included benchmarks {list(benchmark_configs.keys())}"
            )

    if args.machine == "amd" and args.exemode == "jitify":
        raise Exception("Jitify exemode is unavaible on amd")

    cwd = os.getcwd()
    res_dir = pathlib.Path(f"{cwd}/results/")
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.exemode == "aot":
        env_configs = AOTConfig().get_env_configs()
    elif args.exemode == "proteus":
        if not args.proteus_config:
            raise Exception("Missing --proteus-config specification through CLI")
        env_configs = ProteusConfig(**args.proteus_config).get_env_configs()
    elif args.exemode == "jitify":
        env_configs = JitifyConfig().get_env_configs()
    else:
        raise Exception(f"Invalid exemode {args.exemode}")
    proteus_install = args.proteus_path
    assert os.path.exists(proteus_install), f"Error: Proteus install path '{proteus_install}' does not exist!"
    for env in env_configs:
        env["PROTEUS_INSTALL_PATH"] = proteus_install
    experiments = []
    build_command = None
    build_once = False
    # custom toml wide level build command specified
    if "build" in benchmark_configs:
        build_command = benchmark_configs["build"][args.machine]["command"]
        build_once = True

    for benchmark in args.bench if args.bench else benchmark_configs:
        if benchmark == "build":
            continue
        config = benchmark_configs[benchmark]
        experiments.append(
            Executor(
                benchmark,
                Path.cwd() / Path(config[args.machine][args.exemode]["path"]),
                Path(config[args.machine][args.exemode]["exe"]),
                config[args.machine][args.exemode]["args"],
                args.exemode,
                build_command,
                config["inputs"],
                args.compiler,
                args.proteus_path,
                env_configs
            )
        )

    def gather_profiler_results(metrics):
        if args.machine == "amd":
            results_profiler, caching_profiler = e.build_and_run(
                args.reps, Rocprof(metrics, cwd)
            )
        elif args.machine == "nvidia":
            results_profiler, caching_profiler = e.build_and_run(
                args.reps, Nvprof(metrics)
            )
        else:
            raise Exception("Expected amd or nvidia machine")

        # Store the intermediate, benchmark results.
        metrics_suffix = "-metrics" if metrics else ""
        results_profiler.to_csv(
            f"{res_dir}/{args.machine}-{e.benchmark}-{args.exemode}-{args.suffix}-results-profiler{metrics_suffix}.csv"
        )
        caching_profiler.to_csv(
            f"{res_dir}/{args.machine}-{e.benchmark}-{args.exemode}-{args.suffix}-caching-profiler{metrics_suffix}.csv"
        )

    def gather_results():
        results, caching = e.build_and_run(args.reps)
        # Store the intermediate, benchmark results.
        results.to_csv(
            f"{res_dir}/{args.machine}-{e.benchmark}-{args.exemode}-{args.suffix}-results.csv"
        )
        caching.to_csv(
            f"{res_dir}/{args.machine}-{e.benchmark}-{args.exemode}-{args.suffix}-caching.csv"
        )

    # Build, run, and collect results for each experiment as gathered by glob
    # directories. Do profiler runs with and without metrics, and a run without
    # the profiler for end-to-end execution times.
    for e in experiments:
        # Gather results without the profiler.
        if args.profmode == "direct":
            gather_results()
        # Gather results with the machine-specific profiler WITHOUT metrics (gpu
        # counters)
        if args.profmode == "profiler":
            gather_profiler_results(metrics=False)
        # Gather results with the machine-specific profiler WITH metrics (gpu
        # counters).
        if args.profmode == "metrics":
            gather_profiler_results(metrics=True)

    print("Results are stored in ", res_dir)


if __name__ == "__main__":
    main()
