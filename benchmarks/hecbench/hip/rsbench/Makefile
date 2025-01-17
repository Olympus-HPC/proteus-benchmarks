#===============================================================================
# User Options
#===============================================================================

CC ?= ${PROTEUS_CC}
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
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

program = rsbench$(SUFFIX).x

source = \
main.cpp \
simulation.cpp\
io.cpp \
init.cpp \
material.cpp \
utils.cpp

obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS :=

CFLAGS += -std=c++14

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
  -llldCommon -llldELF

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  CFLAGS += -fpass-plugin=$(PROTEUS_PATH)/lib/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,$(PROTEUS_PATH)/lib -L${PROTEUS_PATH}/lib/ -lproteus
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp rsbench.h Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc
edit:
	vim -p $(source) rsbench.h

run:
	./rsbench
