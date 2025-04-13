import argparse
import pandas as pd
from pathlib import Path
import pathlib
import subprocess
import os
import time
import pprint
import re
from itertools import product
import shutil
import tomllib
import functools


@functools.cache
def demangle(potentially_mangled_name):
    result = subprocess.run(
        ["llvm-cxxfilt", potentially_mangled_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demangling error: {result.stderr}")

    return result.stdout.strip()


class Rocprof:
    def __init__(self, metrics):
        self.metrics = metrics
        if metrics:
            metrics_file = f"{__file__}/vis/rocprof-metrics.txt"
            self.command = f"rocprof -i {metrics_file}" + " --timestamp on -o {0} {1}"
        else:
            self.command = "rocprof --timestamp on -o {0} {1}"

    def get_command(self, output, executable):
        return self.command.format(output, executable)

    def parse(self, fn):
        def get_hash(x):
            try:
                hash_pos = 2
                return x.split("$")[hash_pos]
            except IndexError:
                return None

        df = pd.read_csv(fn, sep=",")
        # Rename to match output between rocprof, nvprof.
        df.rename(columns={"KernelName": "Name", "Index": "RunIndex"}, inplace=True)
        df["Duration"] = df["EndNs"] - df["BeginNs"]
        df["Name"] = df["Name"].str.replace(" [clone .kd]", "", regex=False)
        df["Hash"] = df.Name.apply(lambda x: get_hash(x))
        df["Name"] = df.Name.apply(lambda x: demangle(x.split("$")[0]))
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
                return x.split("$")[hash_pos]
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
            df["Kernel"] = df.Kernel.apply(lambda x: demangle(x.split("$")[0]))
            df.rename(columns={"Kernel": "Name"}, inplace=True)
        else:
            df["Hash"] = df.Name.apply(lambda x: get_hash(x))
            df["Name"] = df.Name.apply(lambda x: demangle(x.split("$")[0]))

        return df


class Runner:
    @staticmethod
    def execute_command(cmd, **kwargs):
        print("=> Execute", cmd)

        with subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        ) as p:
            out = ""
            for line in p.stdout:
                print(line, end="")
                out += line

            p.wait()

            err = p.stderr.read()
            if p.returncode:
                raise RuntimeError(f"Failed cmd {cmd}\n=== stderr ===\n{err}")

            return out, err


class Builder:
    def __init__(self, build_path, build_command, clean_command, cc, proteus_path):
        # the build command is meant to be a full bash command to build the
        # benchmark, e.g., `cmake -DCMAKE_BUILD_TYPE=Release --build` or `make
        # benchmark.
        self.build_path = build_path
        self.build_command = build_command
        self.clean_command = clean_command
        self.cc = cc
        self.proteus_path = proteus_path

    def clean(self):
        if os.path.isdir(self.build_path):
            if self.clean_command is not None:
                Runner.execute_command(self.clean_command, cwd=str(self.build_path))

    def build(self, enable_proteus):
        os.makedirs(self.build_path, exist_ok=True)

        self.clean()
        print("BUILD", self.build_path, "enable_proteus?", enable_proteus)

        env = os.environ.copy()
        env["CC"] = self.cc
        env["ENABLE_PROTEUS"] = "yes" if enable_proteus else "no"
        if enable_proteus:
            env["PROTEUS_PATH"] = self.proteus_path

        Executor.build_done = True
        t1 = time.perf_counter()
        print(
            "Build command",
            self.build_command,
            "CC=" + env["CC"],
            "ENABLE_PROTEUS=" + env["ENABLE_PROTEUS"],
            "PROTEUS_PATH=" + env["PROTEUS_PATH"] if enable_proteus else "",
        )
        if not isinstance(self.build_command, list):
            self.build_command = [self.build_command]
        for cmd in self.build_command:
            Runner.execute_command(cmd, env=env, cwd=self.build_path)
        t2 = time.perf_counter()
        self.ctime = t2 - t1


class Executor:
    def __init__(
        self,
        benchmark,
        executable_name,
        extra_args,
        exemode,
        inputs,
        reps,
        profiler,
        env_configs,
        run_path,
        builder,
    ):
        self.benchmark = benchmark
        self.executable_name = executable_name
        self.extra_args = extra_args
        self.exemode = exemode
        self.inputs = inputs
        self.profiler = profiler
        self.reps = reps
        self.env_configs = env_configs
        self.run_path = run_path
        self.builder = builder

    def __str__(self):
        return f"{self.benchmark} {self.run_path} {self.exemode}"

    def run(self):
        results = pd.DataFrame()
        caching = pd.DataFrame()
        assert (
            self.exemode == "aot"
            or self.exemode == "proteus"
            or self.exemode == "jitify"
        ), "Expected aot or proteus or jitify for exemode"

        ctime = self.builder.ctime
        exe_size = (
            Path(f"{self.builder.build_path}/{self.executable_name}").stat().st_size
        )

        for (input_id, args), env, repeat in product(
            self.inputs.items(), self.env_configs, range(0, self.reps)
        ):
            info = f"""
=== Experiment ===
exe: {self.executable_name}
profiler: {self.profiler}
input: {input_id}
args: {args}
extra_args: {self.extra_args}
env: {env}
rep: {repeat}
=== End of Experiment ==="""
            print(info)

            cmd_env = os.environ.copy()
            for k, v in env.items():
                cmd_env[k] = v
            cmd = f"{self.builder.build_path}/{self.executable_name} {args} {self.extra_args}"

            set_launch_bounds = (
                False
                if "PROTEUS_SET_LAUNCH_BOUNDS" not in env
                or env["PROTEUS_SET_LAUNCH_BOUNDS"] == "0"
                else True
            )
            use_stored_cache = (
                False
                if "PROTEUS_USE_STORED_CACHE" not in env
                or env["PROTEUS_USE_STORED_CACHE"] == "0"
                else True
            )
            specialize_args = (
                False
                if "PROTEUS_SPECIALIZE_ARGS" not in env
                or env["PROTEUS_SPECIALIZE_ARGS"] == "0"
                else True
            )

            specialize_dims = (
                False
                if "PROTEUS_SPECIALIZE_DIMS" not in env
                or env["PROTEUS_SPECIALIZE_DIMS"] == "0"
                else True
            )

            # Delete any previous generated Proteus stored cache.
            shutil.rmtree(self.run_path/".proteus", ignore_errors=True)
            if use_stored_cache:
                # Execute a warmup run if using the stored cache to
                # generate the cache files.  CAUTION: We need to create
                # the cache jit binaries right before running.
                # Especially, Proteus launch bounds, runtime args,
                # specialized dims will be baked into the binary so we
                # need a "warmup" run for each setting before taking the
                # measurement.
                Runner.execute_command(
                    cmd,
                    env=cmd_env,
                    cwd=str(self.run_path),
                )

            stats = f"{os.getcwd()}/{self.exemode}-{input_id}-{time.time()}.csv"
            if self.profiler:
                # Execute with profiler on.
                cmd = self.profiler.get_command(stats, cmd)

            t1 = time.perf_counter()
            out, _ = Runner.execute_command(
                cmd,
                env=cmd_env,
                cwd=str(self.run_path),
            )
            t2 = time.perf_counter()

            # Cleanup from a stored cache run, removing cache files.
            cache_size_obj = 0
            cache_size_bc = 0
            if use_stored_cache:
                for file in Path(self.run_path).glob(".proteus/cache-jit-*.o"):
                    # Size in bytes.
                    cache_size_obj += file.stat().st_size
                for file in Path(self.run_path).glob(".proteus/cache-jit-*.bc"):
                    # Size in bytes.
                    cache_size_bc += file.stat().st_size

            # Delete amy previous cache files in the command path.
            shutil.rmtree(self.run_path/".proteus")

            if self.profiler:
                df = self.profiler.parse(stats)
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
                if isinstance(self.profiler, Nvprof):
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


class Experiment:
    def __init__(self, builder, executor):
        self.builder = builder
        self.executor = executor


class ResultsCollector:
    def __init__(
        self,
        executor,
        machine,
        results_dir,
    ):
        self.executor = executor
        self.machine = machine
        self.results_dir = results_dir

    def gather_results(self):
        results, caching = self.executor.run()

        if self.executor.profiler:
            suffix = "profiler"
            suffix += "-metrics" if self.executor.profiler.metrics else ""
        else:
            suffix = "direct"
        results.to_csv(
            f"{self.results_dir}/{self.machine}-{self.executor.benchmark}-{self.executor.exemode}-results-{suffix}.csv"
        )
        caching.to_csv(
            f"{self.results_dir}/{self.machine}-{self.executor.benchmark}-{self.executor.exemode}-caching-{suffix}.csv"
        )

        results.to_csv(
            f"{self.results_dir}/{self.machine}-{self.executor.benchmark}-{self.executor.exemode}-results-{suffix}.csv"
        )
        caching.to_csv(
            f"{self.results_dir}/{self.machine}-{self.executor.benchmark}-{self.executor.exemode}-caching-{suffix}.csv"
        )


def get_profiler(machine, profmode):
    if profmode == "direct":
        return None
    elif profmode == "profiler" or profmode == "metrics":
        metrics_flag = True if profmode == "metrics" else False
        if machine == "amd":
            return Rocprof(metrics=metrics_flag)
        elif machine == "nvidia":
            return Nvprof(metrics=metrics_flag)
        else:
            raise Exception(f"Machine {machine} is not supported.")
    else:
        raise Exception(f"Profiling mode {profmode} is not supported.")


def check_valid_proteus_env(env_configs):
    valid_keys = (
        "PROTEUS_USE_STORED_CACHE",
        "PROTEUS_SET_LAUNCH_BOUNDS",
        "PROTEUS_SPECIALIZE_ARGS",
        "PROTEUS_SPECIALIZE_DIMS",
    )

    for env in env_configs:
        for key, value in env.items():
            if key not in valid_keys:
                raise Exception(f"Invalid key {key} not in {valid_keys}")

            if not all([o in ["0", "1"] for o in value]):
                raise Exception(f"Expected values 0 or 1 for opt {key}, value: {value}")


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
        "-c",
        "--compiler",
        help="path to the compiler executable",
    )
    parser.add_argument(
        "-j",
        "--proteus-path",
        help="path to proteus install directory",
    )
    parser.add_argument(
        "-x",
        "--exemode",
        help="execution mode",
        choices=("aot", "proteus", "jitify"),
    )
    parser.add_argument(
        "-m",
        "--machine",
        help="the machine running on: amd|nvidia",
        choices=("amd", "nvidia"),
    )
    parser.add_argument(
        "-r",
        "--reps",
        help="number of repeats per experiment",
        type=int,
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
        "--runconfig",
        help="runtime configuration",
    )
    parser.add_argument(
        "--results-dir",
        help="the directory to store results",
    )
    args = parser.parse_args()

    with open(args.toml, "rb") as f:
        benchmark_group_configs = tomllib.load(f)
        assert len(benchmark_group_configs.keys()) == 1, (
            "Expected single, top-level key for the benchmark group"
        )
        benchmark_group = list(benchmark_group_configs)[0]
        benchmark_configs = benchmark_group_configs[benchmark_group]
        group_config = benchmark_configs.pop("config", None)

    if args.list:
        pprint.pprint(benchmark_group_configs)
        return

    for bench in args.bench:
        if bench not in benchmark_configs.keys():
            raise Exception(
                f"{bench} not in included benchmarks {list(benchmark_configs.keys())}"
            )

    if args.machine == "amd" and args.exemode == "jitify":
        raise Exception("Jitify exemode is unavaible on amd")

    if args.compiler is None:
        raise Exception("Compiler executable not specified")

    if args.reps is None:
        raise Exception("Provide number of repetitions per experiment, -r/--reps")

    if args.results_dir is None:
        raise Exception("Provide an output results directory, --results-dir")

    results_dir = pathlib.Path(f"{args.results_dir}").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_once = group_config["build_once"]
    except KeyError:
        build_once = False

    try:
        build_command = group_config["build"][args.machine][args.exemode]["command"]
    except KeyError:
        try:
            build_command = group_config["build"][args.machine]["command"]
        except KeyError:
            raise Exception("Build instructions are missing")

    try:
        clean_command = group_config["build"][args.machine]["clean"]["command"]
    except KeyError:
        clean_command = None

    with open(args.runconfig, "rb") as f:
        runconfigs = tomllib.load(f)

    if not runconfigs[args.exemode]:
        raise Exception("Runconfig is empty")

    if args.exemode == "proteus":
        for runconfig in runconfigs[args.exemode]:
            check_valid_proteus_env(runconfig["env"])

    experiments = []
    builders = []
    for benchmark in args.bench if args.bench else benchmark_configs:
        config = benchmark_configs[benchmark]
        try:
            extra_args = config["args"]
        except KeyError:
            extra_args = ""

        try:
            extra_args = (
                extra_args + " " + group_config[args.machine][args.exemode]["args"]
            )
        except KeyError:
            pass

        try:
            build_path = Path.cwd() / Path(
                group_config["build"][args.machine][args.exemode]["path"]
            )
        except KeyError:
            try:
                build_path = Path.cwd() / Path(
                    config[args.machine][args.exemode]["path"]
                )
            except KeyError:
                build_path = Path.cwd() / Path(group_config["path"])

        try:
            run_path = Path.cwd() / Path(config[args.machine][args.exemode]["path"])
        except KeyError:
            try:
                run_path = Path.cwd() / Path(config["path"])
            except KeyError:
                run_path = Path.cwd() / Path(group_config["path"])

        try:
            exe = Path(config[args.machine][args.exemode]["exe"])
        except KeyError:
            exe = Path(group_config["exe"])

        if build_once:
            if not builders:
                builder = Builder(
                    build_path,
                    build_command,
                    clean_command,
                    args.compiler,
                    args.proteus_path,
                )
                builders.append(builder)
        else:
            builder = Builder(
                build_path,
                build_command,
                clean_command,
                args.compiler,
                args.proteus_path,
            )
            builders.append(builder)

        for runconfig in runconfigs[args.exemode]:
            executor = Executor(
                benchmark,
                exe,
                extra_args,
                args.exemode,
                config["inputs"],
                args.reps,
                get_profiler(args.machine, runconfig["profmode"]),
                runconfig["env"],
                run_path,
                builder,
            )

            experiments.append(Experiment(builder, executor))

    def build_and_run_experiments(experiments):
        # Build, run, and collect results for each experiment and profiling mode.
        for builder in builders:
            builder.build(args.exemode == "proteus")
            print(
                "=> Built",
                builder.build_path,
                "ctime",
                builder.ctime,
                "build_once?",
                build_once,
            )
        print("=> BUILD DONE")

        for e in experiments:
            ResultsCollector(
                e.executor,
                args.machine,
                results_dir,
            ).gather_results()

    build_and_run_experiments(experiments)
    print("Results are stored in ", results_dir)


if __name__ == "__main__":
    main()
