#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
# Compiler can be set below, or via environment variable
CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH  ?= /path/to/proteus/install
ENABLE_PROTEUS ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = "-proteus"
else
  SUFFIX = "-aot"
endif

program = sw4ck$(SUFFIX).x

source = main.cu curvilinear4sg.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall
#CFLAGS := --offload-arch=gfx90a:sramecc+:xnack- $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
          -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
          -Wl,--start-group \
          $(shell ls ${ROCM_PATH}/llvm/lib/libclang*.a) \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
          -Wl,--end-group \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
          -llldCommon -llldELF
endif

#===============================================================================
# Targets to Build
#===============================================================================
$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	$(CC) $(CFLAGS) -x hip  -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc

run: $(program)
	$(LAUNCHER) ./$(program) ../sw4ck-cuda/sw4ck.in 100
