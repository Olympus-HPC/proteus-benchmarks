GPU_TYPE=$1
if [ $# -lt 1 ] || [ -z "${GPU_TYPE}" ];
then
    echo "Usage: emit-ablation-studies.sh  <either amd/nvidia>"
    exit 0
fi

export PLOTS_DIR=plots
export RESULTS_DIR=results

python vis/plot-bar-kernel-speedup-ablation.py --dir ${RESULTS_DIR} --plot-dir ${PLOTS_DIR} -m ${GPU_TYPE} -f png --plot-title ${GPU_TYPE}