# MPI build. On JED:
#   module load gcc openmpi hdf5
# After `module load hdf5`, $HDF5_ROOT is set automatically.

CXX      ?= mpicxx
LD       := $(CXX)
HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)
CXXFLAGS += -O3 -Wall -Wextra -Werror -pedantic -std=c++11 $(HDF5_INC)
LDFLAGS  += $(CXXFLAGS) $(HDF5_LIB) -lhdf5 -lm

OBJS = main.o lbm.o output.o

all: lbm_mpi

lbm_mpi: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_mpi *.o *~

distclean: clean
	rm -rf out/ probe.csv

.PHONY: all clean distclean
