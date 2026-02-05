CC = g++

CFLAGS = -Wall -O3 -march=armv8-a+sve2 -fopenmp -fno-tree-vectorize

LDFLAGS = -lm -fopenmp

###### EPAC VEC ######
# CC = clang
# CFLAGS = -Wall -Ofast -ffast-math -march=rv64g -mcpu=avispado -mepi -fno-vectorize -Rpass=loop-vectorize -DEPI -DEPI_EXT_07 -DINDEX64 -DALIGN_TO=1024 # -fopenmp-simd -Rpass-analysis=loop-vectorize
# LDFLAGS = -lm

###### PETSC ######
# PETSC_DIR = $(HOME)/petsc-dare
# PETSC_ARCH = arch-linux-c-vec
# PETSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -DPETSC
# PETSC_LIB = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc # -lmpi

###### MKL #######
# MKL_INC = -I$(MKLROOT)/include -DMKL
# MKL_LIB = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core

EXE = spmv_power spmm_bench mtx_to_bin # petsc_power petsc_ksp
OBJ = mtx.o coo.o csr.o csr_numa.o csr_epi.o csr_merge.o csr_bal.o ell.o ell_sort.o jds.o ell0.o sell.o sellcs_mv.o sellcs_mv_autovector.o sellcs_mv_kernels_epi.o sellcs_format.o sellcs_analyzer.o radix_sort.o sellcs_utils.o csr_mkl.o petsc.o ginkgo_spmv.o sellp.o acsr.o

all: $(EXE)

spmv_power: spmv_power.o $(OBJ)
	$(CC) $(LDFLAGS) $(PETSC_LIB) $(MKL_LIB) $^ -o $@

spmm_bench: spmm_bench.o $(OBJ)
	$(CC) $(LDFLAGS) $(PETSC_LIB) $(MKL_LIB) $^ -o $@

petsc_power: petsc_power.o $(OBJ)
	$(CC) $(LDFLAGS) $(PETSC_LIB) $(MKL_LIB) $^ -o $@

petsc_ksp: petsc_ksp.o $(OBJ)
	$(CC) $(LDFLAGS) $(PETSC_LIB) $(MKL_LIB) $^ -o $@

mtx_to_bin: mtx_to_bin.o mtx.o
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c *.h
	$(CC) $(CFLAGS) $(PETSC_INC) $(MKL_INC) -c $<

clean:
	rm -fv $(EXE) *.o *.i *.s tags

tags: *.h *.c
	ctags --languages=C,C++ --exclude=$(PETSC_DIR)/include/petsc/finclude/*.h -R $(PETSC_DIR) *.h *.c
