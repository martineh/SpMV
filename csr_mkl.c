#include "spmv.h"

#ifdef MKL

void mult_csr_mkl(struct csr_mkl *csr, double *x, double *y)
{
    // mkl_cspblas_dcsrgemv("N", csr->rows, csr->A, csr->i, csr->j, x, y);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr->matrix, csr->descr, x, 0.0, y);
}

struct csr_mkl *create_csr_mkl(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct csr_mkl *csr = SPMV_ALLOC(struct csr_mkl, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;
    csr->i = SPMV_ALLOC(int, rows + 1);
    csr->j = SPMV_ALLOC(int, nnz);
    csr->A = SPMV_ALLOC(double, nnz);
    csr->memusage = sizeof(int) * (rows + 1) + (sizeof(int) + sizeof(double)) * nnz;

    sort_mtx(nnz, mtx, by_row_mtx);

    csr->i[0] = 0;
    for (int l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) {
            csr->j[l] = mtx[l].j;
            csr->A[l] = mtx[l].a;
            l++;
        }
        csr->i[k + 1] = l;
    }

    csr->descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_create_csr(&csr->matrix, SPARSE_INDEX_BASE_ZERO, rows, columns, csr->i, csr->i + 1, csr->j, csr->A);
    mkl_sparse_set_mv_hint(csr->matrix, SPARSE_OPERATION_NON_TRANSPOSE, csr->descr, 1000000);
    mkl_sparse_optimize(csr->matrix);
    return csr;
}

#endif
