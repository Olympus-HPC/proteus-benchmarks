#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
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

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DMKX=4 -std=c++14 -Wall -I./

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
  -llldCommon -llldELF

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
  CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu kernel.h \
          spt.h \
          constants.h utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 10
