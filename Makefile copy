# Compiler
CHPL ?= chpl

# Directories
SRC_DIR := src
BIN_DIR := bin

# Program name (without .chpl)
PROGRAM := main

# Source and target
SRC := $(wildcard $(SRC_DIR)/*.chpl)
TARGET := $(BIN_DIR)/$(PROGRAM)

# Default rule
all: $(TARGET)

# Module
CGNS_MOD_DIR := /apps/partage-elns/1_NhanT/1_CHAMPS_DEV/champs/EXT_LIBS
CGNS_MOD_DIR_SRC := /apps/partage-elns/1_NhanT/1_CHAMPS_DEV/champs/EXT_LIBS/src
COMMON_MOD_DIR := /apps/partage-elns/1_NhanT/1_CHAMPS_DEV/champs/common/src


HDF5_LIB := -L$(HDF5ROOT)/lib -lhdf5
CGNS_LIB := -L$(CGNSROOT)/lib -lcgns -lhdf5
MKL_LIB := -L$(MKLROOT)/lib/intel64 -lmkl_rt -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
METIS_LIB := -L$(METISROOT)/lib -lmetis
MPI_LIB := -L$(MPIROOT)/lib/release -lmpi
PETSC_LIB := -L$(PETSCROOT)/lib -lpetsc

HDF5_INCLUDE := $(HDF5ROOT)/include hdf5.h
MKL_INCLUDE := $(MKLROOT)/include mkl_types.h mkl.h
CGNS_INCLUDE := $(CGNSROOT)/include cgnslib.h
METIS_INCLUDE := $(METISROOT)/include metis.h
MPI_INCLUDE := $(MPIROOT)/include mpi.h
PETSCROOT_INCLUDE := $(PETSCROOT)/include petscksp.h petsc.h petscmat.h
PETSCDIR_INCLUDE := $(PETSC_DIR)/include petscksp.h petsc.h petscmat.h

ALL_INCLUDES := -I$(HDF5_INCLUDE) -I$(MKL_INCLUDE) -I$(CGNS_INCLUDE) -I$(METIS_INCLUDE) -I$(PETSCROOT_INCLUDE) -I$(PETSCDIR_INCLUDE) -I$(SLEPCROOT_INCLUDE) -I$(SLEPCDIR_INCLUDE) -I$(MPI_INCLUDE)
ALL_LIBS := $(MKL_LIB) $(HDF5_LIB) $(CGNS_LIB) $(METIS_LIB) $(PETSC_LIB) $(SLEPC_LIB) $(MPI_LIB)


# Compile Chapel program
$(TARGET): $(SRC) | $(BIN_DIR)
	$(CHPL) -M$(CGNS_MOD_DIR) -M$(CGNS_MOD_DIR_SRC) -M$(COMMON_MOD_DIR) $(ALL_INCLUDES) $(ALL_LIBS) $(SRC) -o $(TARGET) --set blasImpl=off --local --fast

# Create bin directory if it does not exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean target
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean