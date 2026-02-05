#include <petsc.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <libgen.h>

#include "spmv.h"

void *matrix = NULL;
void (*mult)(void *, double *, double *);

PetscErrorCode spmv_shell(Mat A, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    double *px, *py;
    PetscCall(VecGetArray(x, &px));
    PetscCall(VecGetArray(y, &py));
    mult(matrix, px, py);
    PetscCall(VecRestoreArray(x, &px));
    PetscCall(VecRestoreArray(y, &py));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_SeqAIJ_csr(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   /* *aa, */ *a_a;
  PetscInt           m = A->rmap->n;
  // const PetscInt    *aj, *ii, *ridx = NULL;
  // PetscInt           n, i;
  // PetscScalar        sum;
  PetscBool          usecprow = a->compressedrow.use;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
  #pragma disjoint(*x, *y, *aa)
#endif

  PetscFunctionBegin;
  if (a->inode.use && a->inode.checked) {
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  PetscCall(MatSeqAIJGetArrayRead(A, &a_a));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  // ii = a->i;
  if (usecprow) { /* use compressed row format */
#if 0
    PetscCall(PetscArrayzero(y, m));
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = 0.0;
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      /* for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; */
      y[*ridx++] = sum;
    }
#else
    PetscFunctionReturn(PETSC_ERR_SUP);
#endif
  } else { /* do not use compressed row format */
#if 0
    for (i = 0; i < m; i++) {
      n   = ii[i + 1] - ii[i];
      aj  = a->j + ii[i];
      aa  = a_a + ii[i];
      sum = 0.0;
      PetscSparseDensePlusDot(sum, x, aa, aj, n);
      y[i] = sum;
    }
#else
    struct csr_epi matrix;
    matrix.rows = m;
    matrix.columns= m;
    matrix.i = a->i;
    matrix.j = a->j;
    matrix.A = (double *)a_a;
    mult_csr_epi(&matrix, (double *)x, y);
#endif
  }
  PetscCall(PetscLogFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a_a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
    int rows, columns, nnz;
    struct mtx *coo;

    // get parallel configuration
    int np = 1;
    int max_threads = omp_get_max_threads();
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    MPI_Comm_size(PETSC_COMM_WORLD, &np);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    /* #ifdef MPI_Bcast
    #undef MPI_Bcast // PETSc redefines things that should not be redefined
    #endif */

    // create matrix or load from file
    char mtx_file[PATH_MAX];
    PetscBool set;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mtx", mtx_file, sizeof(mtx_file), &set));
    if (!set) {
        printf("MTX file missing (use -mtx flag)\n");
        goto petsc_finalize;
    } else {
        load_mtx(mtx_file, &rows, &columns, &nnz, &coo);
        if (rows != columns) {
            printf("Matrix is not square\n");
            goto petsc_finalize;
        }
    }

    // build sparse matrix
    Vec x, y;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, columns));
    PetscCall(VecSetSizes(y, PETSC_DECIDE, rows));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetFromOptions(y));
    long m, n;
    PetscCall(VecGetLocalSize(x,&n));
    PetscCall(VecGetLocalSize(y,&m));
    Mat A;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, m, n, rows, columns));
    PetscCall(MatSetFromOptions(A));

    // fill matrix
    const char* type;
    PetscCall(MatGetType(A, &type));
    char format[20] = "petsc";
    PetscCall(PetscOptionsGetString(NULL, NULL, "-spmv", format, sizeof(format), NULL));
    if (strcmp(type, MATSHELL) != 0) {
        // PETSc native matrix
        assembly_petsc_mat(A, rows, columns, nnz, coo);
        if (strcmp(type, "seqaij") == 0 && strcmp(format, "csr") == 0) {
            PetscCall(MatSetOperation(A, MATOP_MULT, (void *)MatMult_SeqAIJ_csr));
        } else if (strcmp(type, "mpiaij") == 0 && strcmp(format, "csr") == 0) {
            Mat_MPIAIJ *maij = (Mat_MPIAIJ *)A->data;
            PetscCall(MatSetOperation(maij->A, MATOP_MULT, (void *)MatMult_SeqAIJ_csr));
            PetscCall(MatSetOperation(maij->B, MATOP_MULT, (void *)MatMult_SeqAIJ_csr));
        } else if (strcmp(format, "petsc") != 0) {
            printf("Not implemented yet\n");
            goto petsc_finalize;
        }
    } else {
        if (np > 1) {
            printf("SHELL matrix only works one MPI process\n");
            goto petsc_finalize;
        }
        // Use our SpMV via a SHELL matrix
        MatShellSetOperation(A, MATOP_MULT, (void *)spmv_shell);
        if (strcmp(format, "coo") == 0) {
            matrix = create_coo(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_coo;
        } else if (strcmp(format, "csr") == 0) {
            matrix = create_csr_pad(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr;
        } else if (strcmp(format, "csr_numa") == 0) {
            matrix = create_csr_numa(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_numa;
        } else if (strcmp(format, "csr_epi") == 0) {
            matrix = create_csr_epi(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_epi;
        } else if (strcmp(format, "csr_merge") == 0) {
            matrix = create_csr_merge(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_merge;
        } else if (strcmp(format, "csr_bal") == 0) {
            matrix = create_csr_bal(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_bal;
        } else if (strcmp(format, "ell0") == 0) {
            matrix = create_ell0(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_ell0;
        } else if (strcmp(format, "sell") == 0) {
            #ifdef INDEX64
            struct csr_epi *csr_matrix = create_csr_epi(rows, columns, nnz, coo);
            #else
            struct csr *csr_matrix = create_csr(rows, columns, nnz, 1, coo);
            #endif
            free(coo); // free some memory before building SELL
            matrix = create_sell(csr_matrix);
            mult = mult_sell;
        #ifdef MKL
        } else if (strcmp(format, "csr_mkl") == 0) {
            matrix = create_csr_mkl(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_mkl;
        #endif
        } else if (strcmp(argv[1], "petsc") == 0) {
            // already implemented in PETSc
        } else {
            printf("Unknown matrix format (use -format flag)\n");
            goto petsc_finalize;
        }
    }

    PetscCall(VecSet(x, 1.0));
    PetscCall(VecSet(y, 1.0));
    double norm;
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCall(VecScale(x, 1.0 / norm));
    double res = 0.0;
    if (np == 1) {
        // test result part
        Vec y_ref;
        PetscCall(VecDuplicate(y, &y_ref));
        double *px, *py, *py_ref;
        PetscCall(VecGetArray(x, &px));
        PetscCall(VecGetArray(y_ref, &py_ref));
        mult_mtx(rows, columns, nnz, coo, px, py_ref);
        PetscCall(VecRestoreArray(x, &px));
        PetscCall(MatMult(A, x, y)); // mult(matrix, x, y);
        PetscCall(VecGetArray(y, &py));
        for (int i = 0; i < rows; i++) {
            double d = fabs(py[i] - py_ref[i]);
            // if (d > 1e-5) printf("diff %d %e %e\n", i, py[i], py_ref[i]);
            if (d > res) res = d;
        }
        PetscCall(VecRestoreArray(y, &py));
        PetscCall(VecRestoreArray(y_ref, &py_ref));
    }

    double time_spmv = 0.0;
    double t0 = get_time();
    int it = 0;
    for (;;) {
        it++;
        double t1 = get_time();
        PetscCall(MatMult(A, x, y)); // mult(matrix, x, y);
        double t2 = get_time();
        time_spmv += t2 - t1;
        PetscCall(VecNorm(y, NORM_2, &norm));
        PetscCall(VecScale(y, 1.0 / norm));
        PetscCall(VecCopy(y, x));
        int stop = t2 > 2 + t0 && it >= 10;
        MPI_Bcast(&stop, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (stop) break;
    }


    char *filename = basename(mtx_file);
    char *extension = strrchr(filename, '.');
    if (extension) *extension = 0;
    if (rank == 0) {
        printf("%s_%s %s %i %i %i %i %i %e %e %i %e %e\n",
            format, type, filename, max_threads, np,
            rows, columns, nnz,
            2.0 * nnz * it / time_spmv,
            time_spmv / it,
            it, res, matrix ? (double)((struct csr *)matrix)->memusage : -1);
    }

    petsc_finalize:
    PetscCall(PetscFinalize());
    return 0;
}
