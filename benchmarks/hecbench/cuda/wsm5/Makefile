#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =
PROTEUS_PATH    ?=/path/to/proteus/install
ENABLE_PROTEUS ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = "-proteus"
else
  SUFFIX = "-aot"
endif

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
	  -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl


# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  CFLAGS += -fpass-plugin=$(PROTEUS_PATH)/lib/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) -lLLVM \
	     -Wl,-rpath,$(PROTEUS_PATH)/lib -L$(PROTEUS_PATH)/lib/ -lproteus -lnvptxcompiler_static
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
