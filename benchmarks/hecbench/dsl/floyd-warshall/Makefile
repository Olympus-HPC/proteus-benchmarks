#===============================================================================
# User Options
#===============================================================================

CC        = ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH ?= /path/to/proteus/install

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = "-proteus"

program = floyd-warshall$(SUFFIX).x

source = main.cpp
obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall
LDFLAGS =

ifeq ($(DEBUG),yes)
	CFLAGS += -g
	LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif

CFLAGS += -I${PROTEUS_PATH}/include
LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
    -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
    -Wl,--start-group \
    $(shell ls ${ROCM_PATH}/llvm/lib/libclang*.a) \
    $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
    -Wl,--end-group \
    $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
    -llldCommon -llldELF -lhiprand

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

.PHONY: clean run

clean:
	rm -rf *.x *.o *.ll *.bc .proteus

run: $(program)
	./$(program) 1024 100 16 0
