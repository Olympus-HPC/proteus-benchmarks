#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH  ?=/path/to/proteus/install

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = "-proteus"

program = attention$(SUFFIX).x

source = main.cpp
obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

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

CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so -I${PROTEUS_PATH}/include
LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
    -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
    -Wl,--start-group \
    $(shell ls ${ROCM_PATH}/llvm/lib/libclang*.a) \
    $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
    -Wl,--end-group \
    $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
    -llldCommon -llldELF

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
	./$(program) 65536 2048 100
