#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC       ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = "-jitify"

program = wsm5$(SUFFIX).x

source = main.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DMKX=4 -std=c++14 -Xcompiler -Wall --offload-arch=$(ARCH) --cuda-path=${CUDA_HOME}

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
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu kernel.h spt.h constants.h utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 10
