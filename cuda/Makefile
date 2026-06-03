# CUDA build for izar (Tesla V100, sm_70).
#   module load gcc cuda hdf5
# The hdf5 module sets $HDF5_ROOT, picked up below.

NVCC      := nvcc
ARCH      := -arch=sm_70
HDF5_INC  := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB  := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)
NVCCFLAGS += -O3 $(ARCH) -std=c++14 $(HDF5_INC)
LDFLAGS   += $(HDF5_LIB) -lhdf5 -lm

OBJS = main.o lbm.o output.o

all: lbm_cuda

lbm_cuda: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cc
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f lbm_cuda *.o *~

distclean: clean
	rm -rf out/ probe.csv

.PHONY: all clean distclean
