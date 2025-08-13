MINICONDA_DIR=miniconda3-repro

function setup() {
  mkdir -p ./${MINICONDA_DIR}
  if [ -f ${MINICONDA_DIR}/bin/activate ]; then
    source ./${MINICONDA_DIR}/bin/activate
    return 0
  fi

  export CONDA_ENVS_DIR=${MINICONDA_DIR}/envs
  export CONDA_PKGS_DIR=${MINICONDA_DIR}/pkgs

  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ./${MINICONDA_DIR}/miniconda.sh
  bash ./${MINICONDA_DIR}/miniconda.sh -b -u -p ./${MINICONDA_DIR}
  rm ./${MINICONDA_DIR}/miniconda.sh
  source ./${MINICONDA_DIR}/bin/activate
}

setup