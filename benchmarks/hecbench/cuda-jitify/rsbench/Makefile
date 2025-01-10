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

program = rsbench$(SUFFIX).x

source = \
main.cu \
simulation.cu\
io.cu \
init.cu \
material.cu \
utils.cu

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
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

LDFLAGS +=-lnvptxcompiler_static

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu rsbench.h Makefile
	$(CC) $(CFLAGS) -c  $< -o $@

clean:
	rm -rf *.x *.o

run: $(program)
	$(LAUNCHER) ./$(program) -s large -m event
