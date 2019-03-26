#
# Makefile for building Matrix Multiplication under CUDA
# 

SORKEN = $(shell hostname | grep sorken | wc -c)
SORKEN-COMPUTE = $(shell hostname | grep compute | wc -c)
STAMPEDE = $(shell hostname | grep stampede | wc -c)
AWS = $(shell hostname | grep "ip-" | wc -c)

ifneq ($(SORKEN), 0)
include $(PUB)/Arch/arch.cuda.gnu
else
ifneq ($(SORKEN-COMPUTE), 0)
include $(PUB)/Arch/arch.cuda.gnu
else
ifneq ($(STAMPEDE), 0)
include $(PUB)/Arch/arch.stampede.cuda
else
ifneq ($(AWS), 0)
include $(PUB)/Arch/arch.cuda.gnu
endif
endif
endif
endif

# Set usecache=1 to build the variant that uses cache
# By default, build the code to use shared memory
ifeq ($(usecache), 1)
    C++FLAGS += -DUSE_CACHE
    CFLAGS += -DUSE_CACHE
    NVCCFLAGS += -DUSE_CACHE
endif   

# Set gprof=1 on make command line to compile for gprof profiler
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        NVCCFLAGS += -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif


# Set debug=1 on make command line to keep symbol table info for gdb/cachegrind
ifeq ($(debug), 1)
        NVCCFLAGS += -g -G
        LDFLAGS += -g -G
endif   

# If you want to compile for single precision,
# specify single=1 on the "make" command line
ifeq ($(single), 1)
else
    C++FLAGS += -D_DOUBLE
    CFLAGS += -D_DOUBLE
    NVCCFLAGS += -D_DOUBLE
endif

# Keep arround compiler output files, including the ptx assembler
ifeq ($(keep), 1)
	NVCCFLAGS	+= -keep
	NVCCFLAGS 	+= --ptx
endif

# If you want to use the  CUDA Timer
# specify cuda_timer=1 on the "make" command line
# NVCCFLAGS += -DCUDA_TIMER
ifeq ($(cuda_timer), 1)
	NVCCFLAGS += -DCUDA_TIMER
endif

# Uncomment if you want to report resource requirements (registers etc)
NVCCFLAGS += --ptxas-options=-v
# NVCCFLAGS += --opencc-options -LIST:source=on

# You can set the thread block geometry by specifying bx= and by= on
# the make command line, e.g. make bx=16 by=32
# This feature is useful for the shared memory variant but
# not for the naive variant

# Set up for a default block size of 16 x 16
ifdef bx
	DIM_X = -DBLOCKDIM_X=$(bx)
else
	DIM_X = -DBLOCKDIM_X=16
endif
ifdef by
	DIM_Y = -DBLOCKDIM_Y=$(by)
else
	DIM_Y = -DBLOCKDIM_Y=16
endif

ifdef naive
	NVCCFLAGS += "-DNAIVE"
	CFLAGS += "-DNAIVE"
endif

BLOCKING = $(DIM_X) $(DIM_Y)
NVCCFLAGS += $(BLOCKING)
CFLAGS += $(BLOCKING)
C++FLAGS += $(BLOCKING)

APP=mmpy

OBJECTS = mmpy.o  mmpy_host.o  genMatrix.o cmdLine.o Timer.o utils.o Report.o setGrid.o


$(APP): $(OBJECTS) mmpy_kernel.o
	$(NVCC) -o $@ $(LDFLAGS) $(OBJECTS)  $(LDLIBS)

clean:
	rm -f *.linkinfo *.o  *.vcproj $(APP)

