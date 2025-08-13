# execute this script in the proteus-benchmarks directory
PROTEUS_INSTALL_PATH=$1
PROTEUS_CC=$2
PROTEUS_CXX=$3
MACHINE=$4
BENCHMARKS_TOML=$5
if [ $# -lt 5 ] || [ -z "${PROTEUS_INSTALL_PATH}" ] || [ -z "${PROTEUS_CC}" ] || [ -z "${MACHINE}" ] || [ -z "${BENCHMARKS_TOML}" ];
then
    echo "Usage: generate-results.sh <path to proteus installation> <path to Clang compiler> <either amd/nvidia> <path to toml of benchmarks>"
    exit 0
fi

export PROTEUS_PATH="$(pwd)/$PROTEUS_INSTALL_PATH"
REPS=1

echo "Running AOT"
python driver.py -t ${BENCHMARKS_TOML} \
  -cc ${PROTEUS_CC} \
  -c ${PROTEUS_CXX} -j ${PROTEUS_PATH} -x aot -m ${MACHINE} -r ${REPS} --runconfig presets/aot.toml --results-dir results

echo "Running proteus"
python driver.py -t ${BENCHMARKS_TOML} \
  -cc ${PROTEUS_CC} \
  -c ${PROTEUS_CXX} -j ${PROTEUS_PATH} -x proteus \
  -m ${MACHINE} -r ${REPS} --runconfig presets/proteus.toml --results-dir results
