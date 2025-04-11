LBANN_PATH=../lbann
LBANN_DEP_PATH=../deps

BUILD_NON_HAL=ON
BUILD_HAL=ON

USE_DISTCONV=ON

if [ -z "$ENABLE_PROTEUS" ]; then
    ENABLE_PROTEUS="yes"
fi

if [ "$ENABLE_PROTEUS" = "yes" ]; then
    USE_PROTEUS="ON"
else
    USE_PROTEUS="OFF"
fi

git clone --depth 1 --single-branch --branch v1.4.2 https://github.com/LLNL/Aluminum.git ${LBANN_DEP_PATH}/Aluminum
git clone https://github.com/LLNL/Elemental.git ${LBANN_DEP_PATH}/Hydrogen
pushd ${LBANN_DEP_PATH}/Hydrogen && git checkout 4fa2f41c55d705fc6c4b65aa72e1cb691370f3bc && popd
git clone https://github.com/LBANN/DiHydrogen.git ${LBANN_DEP_PATH}/DiHydrogen
pushd ${LBANN_DEP_PATH}/DiHydrogen && git checkout f072d243af0320e8046ae99d04028f4b6371c52b && popd

echo "Using proteus? ${USE_PROTEUS}"
echo "PROTEUS_PATH: ${PROTEUS_PATH}"

CLUSTER=$(hostname | sed 's/[0-9]*//g')
ROCM_VERSION=$(echo ${ROCM_PATH} | grep --color=no -o "[0-9]\.[0-9]\.[0-9]")

MPI_LIB=default-mpi
case "${CLUSTER}" in
    rzvernal|tioga)
        MPI_LIB=cray-mpich-vod
	RARCH=gfx90a
        ;;
    tuolumne)
	MPI_LIB=cray-mpich-vod
	RARCH=gfx942
	;;
    corona)
        MPI_LIB=openmpi-4.1.2
	RARCH=gfx906
        ;;
    *)
        ;;
esac

INSTALL_TOP=./deps-${CLUSTER}/rocm-${ROCM_VERSION}/${MPI_LIB}

# This is zsh specific.
#typeset -TU CMAKE_PREFIX_PATH cmake_prefix_path
#cmake_prefix_path=(
#    ${INSTALL_TOP}/adiak
#    ${INSTALL_TOP}/caliper
#    ${INSTALL_TOP}/catch2
#    ${INSTALL_TOP}/cereal
#    ${INSTALL_TOP}/clara
#    ${INSTALL_TOP}/cnpy
#    ${INSTALL_TOP}/conduit
#    ${INSTALL_TOP}/hdf5
#    ${INSTALL_TOP}/hipTT
#    ${INSTALL_TOP}/jpeg-turbo
#    ${INSTALL_TOP}/opencv
#    ${INSTALL_TOP}/protobuf
#    ${INSTALL_TOP}/rccl
#    ${INSTALL_TOP}/spdlog
#    ${INSTALL_TOP}/zstr
#    $cmake_prefix_path)

cmake \
    -G Ninja \
    -S $LBANN_PATH/scripts/superbuild \
    -B . \
    \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_TOP} \
    \
    -D CMAKE_C_COMPILER=$(which amdclang) \
    -D CMAKE_CXX_COMPILER=$(which amdclang++) \
    -D CMAKE_Fortran_COMPILER=$(which gfortran) \
    \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_HIP_STANDARD=17 \
    -D CMAKE_HIP_ARCHITECTURES=$RARCH \
    \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
    -D LBANN_SB_DEFAULT_ROCM_OPTS=ON \
    \
    -D LBANN_SB_BUILD_Catch2=${BUILD_NON_HAL} \
    -D LBANN_SB_Catch2_TAG="devel" \
    \
    -D LBANN_SB_BUILD_adiak=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_Caliper=${BUILD_NON_HAL} \
    -D LBANN_SB_adiak_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_Caliper_BUILD_SHARED_LIBS=ON \
    \
    -D LBANN_SB_BUILD_cereal=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_Clara=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_CNPY=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_protobuf=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_spdlog=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_zstr=${BUILD_NON_HAL} \
    \
    -D LBANN_SB_BUILD_Conduit=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_HDF5=${BUILD_NON_HAL} \
    \
    -D LBANN_SB_BUILD_JPEG-TURBO=${BUILD_NON_HAL} \
    -D LBANN_SB_BUILD_OpenCV=${BUILD_NON_HAL} \
    -D LBANN_SB_OpenCV_TAG=4.x \
    \
    -D LBANN_SB_BUILD_RCCL=OFF \
    -D LBANN_SB_FWD_RCCL_CMAKE_CXX_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_RCCL_CMAKE_HIP_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_RCCL_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_RCCL_BUILD_TYPE=Release \
    -D LBANN_SB_RCCL_CXX_COMPILER=$(which hipcc) \
    -D LBANN_SB_FWD_RCCL_CMAKE_HIP_ARCHITECTURES=$RARCH \
    -D LBANN_SB_FWD_RCCL_AMDGPU_TARGETS=$RARCH \
    -D LBANN_SB_FWD_RCCL_GPU_TARGETS=$RARCH \
    \
    -D LBANN_SB_BUILD_Aluminum=${BUILD_HAL} \
    -D LBANN_SB_Aluminum_SOURCE_DIR=${LBANN_DEP_PATH}/Aluminum \
    -D LBANN_SB_Aluminum_CXX_FLAGS="-Og -g3" \
    -D LBANN_SB_Aluminum_HIP_FLAGS="-Og -g3" \
    -D LBANN_SB_FWD_Aluminum_CXX_FLAGS_RELEASE="-Og -g3 -DNDEBUG" \
    -D LBANN_SB_FWD_Aluminum_HIP_FLAGS_RELEASE="-Og -g3 -DNDEBUG" \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_CALIPER=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_NCCL=ON \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_HOST_TRANSFER=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_TESTS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_BENCHMARKS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_THREAD_MULTIPLE=OFF \
    \
    -D LBANN_SB_BUILD_Hydrogen=${BUILD_HAL} \
    -D LBANN_SB_Hydrogen_SOURCE_DIR=${LBANN_DEP_PATH}/Hydrogen \
    -D LBANN_SB_Hydrogen_CXX_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_Hydrogen_HIP_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_FWD_Hydrogen_CMAKE_CXX_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_Hydrogen_CMAKE_HIP_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_HALF=OFF \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_TESTING=ON \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_UNIT_TESTS=OFF \
    \
    -D LBANN_SB_BUILD_DiHydrogen=${BUILD_HAL} \
    -D LBANN_SB_DiHydrogen_SOURCE_DIR=${LBANN_DEP_PATH}/DiHydrogen \
    -D LBANN_SB_DiHydrogen_CXX_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_DiHydrogen_HIP_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_FWD_DiHydrogen_CMAKE_CXX_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_DiHydrogen_CMAKE_HIP_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_DiHydrogen_H2_ENABLE_DISTCONV_LEGACY=${USE_DISTCONV} \
    \
    -D LBANN_SB_BUILD_LBANN=${BUILD_HAL} \
    -D LBANN_SB_LBANN_PREFIX=${PWD}/install \
    -D LBANN_SB_LBANN_SOURCE_DIR=$LBANN_PATH \
    -D LBANN_SB_FWD_LBANN_CMAKE_CXX_STANDARD=17 \
    -D LBANN_SB_FWD_LBANN_cuTT_ROOT=$INSTALL_TOP/hipTT \
    -D LBANN_SB_LBANN_CXX_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_LBANN_HIP_FLAGS="-Wno-deprecated-declarations -fno-omit-frame-pointer" \
    -D LBANN_SB_FWD_LBANN_CMAKE_CXX_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_LBANN_CMAKE_HIP_FLAGS_RELEASE="-Og -DNDEBUG -g3" \
    -D LBANN_SB_FWD_LBANN_CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -D LBANN_SB_LBANN_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_CALIPER=OFF \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_PROTEUS=${USE_PROTEUS} \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_DISTCONV=${USE_DISTCONV} \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_TBINF=OFF \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_UNIT_TESTING=ON \
    -D LBANN_SB_FWD_LBANN_LBANN_DATATYPE=float
