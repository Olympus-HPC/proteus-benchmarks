MPI_FLAGS =

CC?= ${PROTEUS_CC}
PROTEUS_PATH ?= /path/to/proteus/install
ENABLE_PROTEUS ?= no
FLAGS	=
DFLAGS	= $(MPI_FLAGS) -lineinfo
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = "-proteus"
else
  SUFFIX = "-aot"
endif


#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

LINKFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
	-llldCommon -llldELF
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

# release flags
FLAGS += $(RFLAGS)

ifeq ($(ENABLE_PROTEUS),yes)
  FLAGS += -fpass-plugin=$(PROTEUS_PATH)/lib/libProteusPass.so -DENABLE_PROTEUS
  LINKFLAGS += -Wl,-rpath,$(PROTEUS_PATH)/lib -L${PROTEUS_PATH}/lib/ -lproteus
endif

lulesh$(SUFFIX).x: allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o
	$(CC) $(LINKFLAGS) allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o -o $@

allocator$(SUFFIX).o: allocator.cu vector.h
	$(CC) $(FLAGS) allocator.cu -I ./ -c -o $@

lulesh$(SUFFIX).o: lulesh.cu util.h vector.h allocator.h
	$(CC) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o $@

lulesh-comms$(SUFFIX).o: lulesh-comms.cu
	$(CC) $(FLAGS) lulesh-comms.cu -I ./ -c -o $@

lulesh-comms-gpu$(SUFFIX).o: lulesh-comms-gpu.cu
	$(CC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o $@

clean:
	rm -rf *.x *.o xyz.asc regNumList.txt
