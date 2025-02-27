# proteus-benchmarks

## Description

This repository contains a collection of benchmark programs for [proteus](https://github.com/Olympus-HPC/proteus).

The directory structure is:

```
.
├── benchmarks              # Source files of benchmark programs
├── setup                   # Scripts to setup proteus
├── vis                     # Scripts for plotting
├── benchmark.toml          # Benchmark configurations using TOML
├── COPYRIGHT
├── driver.py               # Driver script to run benchmark programs
└── LICENSE
└── README.md
```

## How to use

### Setup proteus
First, setup proteus:
```bash
source setup-proteus-env.sh <env-dir>
```
which uses `conda` to install Clang/LLVM dependencies and builds and installs proteus.

> [!NOTE]
> The setup scripts expects a CUDA/HIP installation. It detects CUDA or HIP using CUDA_PATH/CUDA_HOME or ROCM_PATH respectively. Needed environment modules should be loaded before running the scripts

### Run the driver
Then run the driver script `driver.py` to execute benchmark programs. This is its help output:
```
(proteus)$ python driver.py  --help
usage: driver.py [-h] -t TOML -c COMPILER -j PROTEUS_PATH -x {aot,proteus,jitify} -p {direct,profiler,metrics} -m {amd,nvidia} -r REPS [-l]
                 [-b BENCH [BENCH ...]] [--proteus-config PROTEUS_CONFIG]

Build, run and collect measurements for a benchmark program

options:
  -h, --help            show this help message and exit
  -t TOML, --toml TOML  input toml descriptors for benchmarks
  -c COMPILER, --compiler COMPILER
                        path to the compiler executable
  -j PROTEUS_PATH, --proteus-path PROTEUS_PATH
                        path to proteus install directory
  -x {aot,proteus,jitify}, --exemode {aot,proteus,jitify}
                        execution mode
  -p {direct,profiler,metrics}, --profmode {direct,profiler,metrics}
                        profiling mode
  -m {amd,nvidia}, --machine {amd,nvidia}
                        the machine running on: amd|nvidia
  -r REPS, --reps REPS  number of repeats per experiment
  -l, --list            list available benchmarks and configurations
  -b BENCH [BENCH ...], --bench BENCH [BENCH ...]
                        run a particular benchmark
  --proteus-config PROTEUS_CONFIG
                        proteus env var configuration
```
Most options are self-explanatory.
We explain the trickiest ones.

The option `--profmode` executes a benchmark program: 
1. without using a profiler in `direct` mode to collect end-to-end execution time measurements; or
2. using a machine-specific profiler (i.e., `nvprof` for CUDA and `rocprof` for HIP) in `profiler` mode to collect measurements for a GPU trace; or
3. using a machine-specific profiler and collecting GPU performance counters in `metrics` mode


The option `--proteus-configs` which sets environment variable values to configure proteus execution.
It expects a string representation of a dictionary specifying those env vars.
For example:
```bash
python driver.py -t benchmarks.toml -c hipcc -j $(realpath proteus-env/proteus/build-amd/install) -x proteus -p direct -m amd -r 1 --proteus-config '{"PROTEUS_USE_STORED_CACHE":["0", "1"], "PROTEUS_SET_LAUNCH_BOUNDS":["1"], "PROTEUS_SPECIALIZE_ARGS":["1"], "PROTEUS_SPECIALIZE_DIMS": ["1"]}' -b adam
```

This will run the benchmark program `adam` under `proteus` in `direct` mode, so
without a profiler.
The provided proteus configuration generates the cross-product of all provided values
for the configuration keys.
In this example, it will generate 2 different experiments: (1) with stored
caching disabled and all specializations enabled, and (2) with stored caching
enabled and again all specializations enabled.

The driver stores measurements results in CSV format under a created directory named `results`.

### Run visualization scripts (plotting)

The directory `vis` contains several plotting scripts that plot for various aspects of proteus execution (end-to-end execution time, kernel-only execution time, overheads).
Directing profiling modes (the `--profmode` option of `driver.py`) enable different plots.

Profiling mode `direct` enables:
1. `plot-bar-end2end-speedup.py`, showing end-to-end speedup of proteus over ahead-of-time (AOT) compilation
2. `plot-bar-end2end-speedup-noopt.py`, showing end-to-end speedup (slowdown if < 1) of proteus without JIT optimizations over AOT
3. `plot-bar-compilation-slowdown.py`, showing AOT compilation slowdown due to proteus extensions

Profilimg node `profiler`enables:
1. `plot-bar-kernel-speedup.py`, showing speedup for kernel-only execution time of proteus over AOT

Profiling mode `metrics` enables:
1. `plot-barh-kernel-duration.py`, showing the duration (execution time) of individual kernels
2. `plot-barh-metric.py`, showing results of a specific GPU performance counter or metric

> [!WARNING]
> Plotting scripts for `metrics` are experimental and under development.

## License
proteus-benchmarks is distributed under the terms of the Apache License (Version 2.0) with LLVM Exceptions.

All new contributions must be made under the Apache-2.0 with LLVM Exceptions license.

See LICENSE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

LLNL-CODE-2000857
