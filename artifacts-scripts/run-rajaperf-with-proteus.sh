MACHINE=$1
LLVM_INSTALL_DIR=$2
if [ $# -lt 2 ] || [ -z "${MACHINE}" ] || [ -z "${LLVM_INSTALL_DIR}" ] ;
then
    echo "Usage: run-rajaperf-with-proteus.sh <amd/nvidia> <path to LLVM 18 installation>"
    exit 0
fi

git clone https://github.com/Olympus-HPC/proteus.git
cd proteus
git reset --hard 1ace4ca7c7cfe810065e37adb07a40a088c1e6e6
cd ..

if [[ "$MACHINE" == "amd" ]]; then
  mkdir proteus/build-rocm
  cmake  \
    -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
    -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
    -DPROTEUS_ENABLE_HIP=on \
    -DCMAKE_INSTALL_PREFIX=install-proteus \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
    -DENABLE_TESTS=Off \
    -DPROTEUS_ENABLE_DEBUG=Off \
    -B proteus/build-rocm \
    -S proteus
  cmake --build proteus/build-rocm --target install -- -j
elif [[ "$MACHINE" == "nvidia" ]]; then
  mkdir proteus/build-cuda
  cmake  \
    -DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
    -DPROTEUS_ENABLE_CUDA=on \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_C_COMPILER="$LLVM_INSTALL_DIR/bin/clang" \
    -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
    -DCMAKE_CUDA_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
    -DPROTEUS_LINK_SHARED_LLVM=on \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
    -DENABLE_TESTS=Off \
    -DCMAKE_INSTALL_PREFIX=install-proteus \
    -B proteus/build-cuda \
    -S proteus
  cmake --build proteus/build-cuda --target install -- -j
else
  echo "Usage: /run-rajaperf-with-proteus.sh <amd/nvidia> <path to LLVM 17 or 18 installation>"
  exit 1
fi

echo "Built and installed Proteus"
echo "Setting up conda env"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=1
source setup/install-miniconda3.sh
conda create -y -n proteus-repro -c conda-forge \
      python=3.12 cmake=3.24.3 pandas cxxfilt matplotlib

source miniconda3-repro/etc/profile.d/conda.sh
conda activate proteus-repro

# we need to patch BLT's CMake for RAJAPerf to work with clang cuda
# This change needs to be upstreamed.
cd benchmarks/RAJAPerf/blt
git apply ../../../blt_clangcuda.patch  --ignore-space-change --ignore-whitespace
cd ../../../

./artifacts-scripts/generate-results.sh install-proteus "$LLVM_INSTALL_DIR/bin/clang" "$LLVM_INSTALL_DIR/bin/clang++" $MACHINE rajaperf.toml
./artifacts-scripts/emit-ablation-studies.sh $MACHINE
