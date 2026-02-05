#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "spmv.h"
#include "sellp.h"
#include "acsr.h"

double vec_norm(int n, double *x)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

int main(int argc, char *argv[])
{
    int rows, columns, nnz;
    struct mtx *coo;

    // get parallel configuration
    int np = 1;
    int max_threads = omp_get_max_threads();
    #ifdef PETSC
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    MPI_Comm_size(PETSC_COMM_WORLD, &np);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    #ifdef MPI_Bcast
    #undef MPI_Bcast // PETSc redefines things that should not be redefined
    #endif
    argc = 3; // ugly hack needed for PETSc arguments
    #endif

    // create matrix or load from file
    if (argc == 3) {
        char *p = strrchr(argv[2], '.');
        if (p != NULL && strcmp(p, ".bin") == 0) {
            load_bin(argv[2], &rows, &columns, &nnz, &coo);
        } else if (p != NULL && strcmp(p, ".mtx") == 0) {
            load_mtx(argv[2], &rows, &columns, &nnz, &coo);
        } else {
            printf("Unkown file extension\n");
            return 1;
        }
        if (rows != columns) {
            printf("Matrix is not square\n");
            return 1;
        }
    } else if (argc == 5) {
        rows = columns = atof(argv[3]);
        int width = atoi(argv[4]);
        if (strcmp(argv[2], "band") == 0) {
            create_band(rows, width, &nnz, &coo);
        } else if (strcmp(argv[2], "arrow") == 0) {
            create_arrow(rows, width, &nnz, &coo);
        } else {
            printf("Unknown matrix\n");
            return 1;
        }
    } else {
        printf("Usage: spmv_power [coo|csr|csr_numa|csr_epi|sell|csr_merge|csr_bal] <mtx file>\n");
        printf("       spmv_power [coo|csr|csr_numa|csr_epi|sell|csr_merge|csr_bal] [band|arrow] <size> <width>\n");
        return 1;
    }

    // test result part 1
    double *x = SPMV_ALLOC(double, columns);
    double *y = SPMV_ALLOC(double, rows);
    double *y_ref = SPMV_ALLOC(double, rows);
    if (strcmp(argv[1], "csr_numa") == 0) {
        // parallel initialization
        for (int i = 0; i < columns; i++) x[i] = y[i] = 1.0;
    } else {
        // sequential initialization
        for (int i = 0; i < columns; i++) x[i] = y[i] = 1.0;
    }
    double norm = vec_norm(columns, x);
    for (int i = 0; i < columns; i++) x[i] /= norm;
    mult_mtx(rows, columns, nnz, coo, x, y_ref);

    // build sparse matrix
    void *matrix;
    void (*mult)(void *, double *, double *);

    if (strcmp(argv[1], "coo") == 0) {
        matrix = create_coo(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_coo;
    } else if (strcmp(argv[1], "csr") == 0) {
        matrix = create_csr_pad(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr;
    } else if (strcmp(argv[1], "csr_base") == 0) {
        matrix = create_csr_pad(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_base;
    } else if (strcmp(argv[1], "acsr") == 0) {
        struct csr * mat_csr = create_csr_pad(rows, columns, nnz, coo);
	matrix = create_acsr(rows, columns, nnz, mat_csr->A, mat_csr->j, mat_csr->i);
        mult = (void (*)(void *, double *, double *))mult_acsr;
    } else if (strcmp(argv[1], "csr_numa") == 0) {
        matrix = create_csr_numa(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_numa;
    } else if (strcmp(argv[1], "csr_epi") == 0) {
        matrix = create_csr_epi(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_epi;
    /* } else if (strcmp(argv[1], "pcsr") == 0) {
        matrix = create_pcsr(rows, columns, nnz, coo, 64 * 1024);
        mult = (void (*)(void *, double *, double *))mult_pcsr;
    } else if (strcmp(argv[1], "csri") == 0) {
        matrix = create_csri(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csri; */
    } else if (strcmp(argv[1], "csr_merge") == 0) {
        matrix = create_csr_merge(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_merge;
    } else if (strcmp(argv[1], "csr_bal") == 0) {
        matrix = create_csr_bal(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_bal;
    } else if (strcmp(argv[1], "ell") == 0) {
        matrix = create_ell(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_ell;
    } else if (strcmp(argv[1], "ell0") == 0) {
        matrix = create_ell0(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_ell0;
    } else if (strcmp(argv[1], "sellp") == 0) {
        struct csr * mat_csr = create_csr_pad(rows, columns, nnz, coo);
        matrix = create_sellp(rows, columns, nnz, mat_csr->i, mat_csr->j, mat_csr->A);
        mult   = (void (*)(void *, double *, double *))mult_sellp;
    } else if (strcmp(argv[1], "sell") == 0) {
        #ifdef INDEX64
        struct csr_epi *csr_matrix = create_csr_epi(rows, columns, nnz, coo);
        #else
        struct csr *csr_matrix = create_csr(rows, columns, nnz, 1, coo);
        #endif
        free(coo); // free some memory before building SELL
        matrix = create_sell(csr_matrix);
        mult = mult_sell;
    #ifdef MKL
    } else if (strcmp(argv[1], "csr_mkl") == 0) {
        matrix = create_csr_mkl(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_csr_mkl;
    #endif
    #ifdef PETSC
    } else if (strcmp(argv[1], "petsc") == 0) {
        matrix = create_petsc(rows, columns, nnz, coo);
        mult = (void (*)(void *, double *, double *))mult_petsc;
    #endif
    } else {
        printf("Unknown matrix format\n");
        return 1;
    }

    mult(matrix, x, y);

    double res = 0.0;
    for (int i = 0; i < rows; i++) {
        double d = fabs(y[i] - y_ref[i]);
        // if (d > 1e-5) printf("diff %d %e %e\n", i, y[i], y_ref[i]);
        if (d > res) res = d;
    }

    double time_spmv = 0.0;
    double t0 = get_time();
    int it = 0;
    for (;;) {
        it++;
        double t1 = get_time();
        mult(matrix, x, y);
        double t2 = get_time();
        time_spmv += t2 - t1;
        norm = vec_norm(columns, y);
        for (int i = 0; i < columns; i++) x[i] = y[i] / norm;
        int stop = t2 > 2 + t0 && it >= 10;
        #ifdef PETSC
        MPI_Bcast(&stop, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        #endif
        if (stop) break;
    }

    char *filename = basename(argv[2]);
    char *extension = strrchr(filename, '.');
    if (extension) *extension = 0;
    #ifdef PETSC
    if (rank == 0)
    #endif
    printf("%s %s %i %i %i %i %i %.4f %e %i %e %e\n",
            argv[1], filename, max_threads, np,
            rows, columns, nnz,
            2.0 * nnz * it / time_spmv / 1000000000,
            time_spmv / it,
            it, res, (double)((struct csr *)matrix)->memusage);

    #ifdef PETSC
    PetscCall(PetscFinalize());
    #endif
    return 0;
}


