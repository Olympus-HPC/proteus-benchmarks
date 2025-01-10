#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = "-jitify"

program = adam$(SUFFIX).x

source = main.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME}

# Linker Flags
LDFLAGS = -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 \
	  -lnvrtc -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3# -ffast-math
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.x *.o

run: $(program)
	$(LAUNCHER) ./$(program) 10000 200 100
