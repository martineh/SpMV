#include "spmv.h"

void mult_coo(struct coo *coo, double *x, double *y)
{
    for (int i = 0; i < coo->rows; i++) y[i] = 0.0;
    double s = 0.0;
    for (int k = 0; k < coo->nnz; k++) {
        s += x[coo->j[k]] * coo->A[k];
        if (k == coo->nnz - 1 || coo->i[k] != coo->i[k + 1]) {
            y[coo->i[k]] += s;
            s = 0.0;
        }
    }
}

struct coo *create_coo(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct coo *coo = SPMV_ALLOC(struct coo, 1);
    coo->rows = rows;
    coo->columns = columns;
    coo->nnz = nnz;
    coo->i = SPMV_ALLOC(int, nnz);
    coo->j = SPMV_ALLOC(int, nnz);
    coo->A = SPMV_ALLOC(double, nnz);
    coo->memusage = (2 * sizeof(int) + sizeof(double)) * nnz;

    sort_mtx(nnz, mtx, by_row_mtx);

    for (int k = 0; k < nnz; k++) {
        coo->i[k] = mtx[k].i;
        coo->j[k] = mtx[k].j;
        coo->A[k] = mtx[k].a;
    }

    return coo;
}

void free_coo(struct coo *coo) {
    SPMV_FREE(coo->i);
    SPMV_FREE(coo->j);
    SPMV_FREE(coo->A);
    SPMV_FREE(coo);
}
