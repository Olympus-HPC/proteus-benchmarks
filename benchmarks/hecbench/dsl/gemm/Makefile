#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH  ?= /path/to/proteus/install

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = -proteus

program = gemm$(SUFFIX).x

source = main.cpp
obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

OFFLOAD_ARCH ?= gfx942
HIPFLAGS := --offload-arch=${OFFLOAD_ARCH}
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -DPROTEUS_ENABLE_HIP -I${PROTEUS_PATH}/include -I${ROCM_PATH}/llvm/include/ -I${ROCM_PATH}/include $(shell ${ROCM_PATH}/llvm/bin/llvm-config --cxxflags) -fexceptions $(HIPFLAGS)

LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
	-llldCommon -llldELF -lamdhip64 -lhiprtc -lhiprtc-builtins -Wl,-rpath,${ROCM_PATH}/lib

ifeq ($(DEBUG),yes)
	CFLAGS += -g
	LDFLAGS  += -g
endif

ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif

LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc .proteus

run: $(program)
	./$(program) --no-verify --N 8192 --kernel jit_regtiled
