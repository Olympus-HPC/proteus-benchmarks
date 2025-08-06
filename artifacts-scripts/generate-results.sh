PROTEUS_INSTALL_PATH=$1
PROTEUS_CC=$2
MACHINE=$3
BENCHMARKS_TOML=$4
if [ $# -lt 2 ] || [ -z "${LLVM_INSTALL_DIR}" ] || [ -z "${CUDA_VERSION}" ];
then
    echo "Usage: source generate-results.sh <path to proteus installation> <path to Clang compiler> <machine--either amd/nvidia> <path to toml of benchmarks>"
    return 0
fi

export PROTEUS_PATH=$PROTEUS_INSTALL_PATH
REPS=1


echo "Running AOT"
python driver.py -t ${BENCHMARKS_TOML} \
  -c ${PROTEUS_CC} -j ${PROTEUS_INSTALL_PATH} -x aot -m ${MACHINE} -r ${REPS} --runconfig presets/aot.toml --results-dir results

echo "Running proteus"
python driver.py -t ${BENCHMARKS_TOML} \
  -c ${PROTEUS_CC} -j ${PROTEUS_INSTALL_PATH} -x proteus \
  -m ${MACHINE} -r ${REPS} --runconfig presets/proteus.toml --results-dir results
