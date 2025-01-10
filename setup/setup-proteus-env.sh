#!/bin/bash

ROOT_DIR="${PWD}"

function error_popd() {
  echo "Error occured: $1"
  cd ${ROOT_DIR}
}

function setup() {
  PROTEUS_ENV_DIR="$1"

  if [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
      echo "=> Detected CUDA"
      EXTRA_PKGS="clang=17.0.5 clangxx=17.0.5 llvmdev=17.0.5 lit=17.0.5"
      CMAKE_SCRIPT=cmake-nvidia.sh
      VENDOR=nvidia
      export PROTEUS_CC=${CONDA_PREFIX}/bin/clang++
  elif [ -n "$ROCM_PATH" ]; then
      echo "=> Detected ROCm"
      EXTRA_PKGS="llvmdev=17.0.5 lit=17.0.5"
      CMAKE_SCRIPT=cmake-amd.sh
      VENDOR=amd
      export PROTEUS_CC=hipcc
  else
    echo "=> Failed to detect CUDA or ROCm installation. Check your environment and try again."
    return 1
  fi

  mkdir -p "$PROTEUS_ENV_DIR"
  cd "$PROTEUS_ENV_DIR"

  # Install miniconda to setup the environment.
  source ${ROOT_DIR}/setup/install-miniconda3.sh || { error_popd $?; return 1; }

  if conda activate proteus &> /dev/null; then
    echo "Environment proteus already exists and activated"
  else
    echo "Creating environment proteus..."
    conda create -y -n proteus -c conda-forge \
      python=3.12 cmake=3.24.3 pandas cxxfilt matplotlib ${EXTRA_PKGS} || { error_popd $?; return 1; }

    conda activate proteus || { error_popd $?; return 1; }
  fi

  # Fix to expose the FileCheck executable, needed for building Proteus.
  if [ ! -f ${CONDA_PREFIX}/bin/FileCheck ]; then
    ln -s ${CONDA_PREFIX}/libexec/llvm/FileCheck ${CONDA_PREFIX}/bin
  fi

  # Clone and build Proteus.
  rm -rf proteus
  git clone --single-branch --branch main --depth 1 https://github.com/Olympus-HPC/proteus.git
  cd proteus
  bash ${ROOT_DIR}/setup/${CMAKE_SCRIPT} || { error_popd $?; return 1; }


  cd build-${VENDOR}
  make -j install || { error_popd $?; return 1; }
  export PROTEUS_INSTALL_PATH=${PWD}/install
}

# Check if script is sourced.
if [ -n "$BASH_SOURCE" ] && [ "$BASH_SOURCE" = "$0" ]; then
  echo "This script MUST BE sourced (from the root directory), do: source $0"
  exit 1
fi

# Check if the correct number of arguments are provided.
if [ $# -ne 1 ]; then
  echo "Usage: source setup/setup-proteus-env.sh <env-dir>"
  return 1
fi

setup "$1"

cd ${ROOT_DIR}