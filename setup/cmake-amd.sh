# NOTE: load any needed modules for ROCm

LLVM_INSTALL_DIR=${ROCM_PATH}/llvm

rm -rf build-amd
mkdir build-amd
pushd build-amd

cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DCMAKE_INSTALL_PREFIX=install \
-DENABLE_HIP=on \
-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on

popd
