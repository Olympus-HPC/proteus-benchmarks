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

program = conv3d$(SUFFIX).x

source = main.cpp
obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
OFFLOAD_ARCH ?= gfx942
HIPFLAGS := --offload-arch=${OFFLOAD_ARCH}
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -DPROTEUS_ENABLE_HIP -I${PROTEUS_PATH}/include -I${ROCM_PATH}/llvm/include/ -I${ROCM_PATH}/include $(shell ${ROCM_PATH}/llvm/bin/llvm-config --cxxflags) -fexceptions $(HIPFLAGS)

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
	-llldCommon -llldELF -lamdhip64 -lhiprtc -lhiprtc-builtins -Wl,-rpath,${ROCM_PATH}/lib

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g
	LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif


# Always link against Proteus runtime for DSL builds
LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

clean:
	rm -rf *.x *.o .proteus

run: $(program)
	./$(program) 32 96 256 26 26 5 100
