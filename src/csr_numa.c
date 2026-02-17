#include "spmv.h"

void mult_csr_numa(struct csr *csr, double *x, double *y)
{
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < csr->rows; k++) {
        double s = 0.0;
#if 0
        #pragma omp simd
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            s += x[csr->j[l]] * csr->A[l];
        }
#else
        int64_t *j = csr->j + csr->i[k];
        double *A = csr->A + csr->i[k];
        #pragma omp simd
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            s += x[*j++] * *A++;
        }
#endif
        y[k] = s;
    }
}

struct csr *create_csr_numa(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct csr *csr = SPMV_ALLOC(struct csr, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;
    csr->i = SPMV_ALLOC(int64_t, rows + 1);
    csr->j = SPMV_ALLOC(int64_t, nnz);
    csr->A = SPMV_ALLOC(double, nnz);
    csr->memusage = sizeof(int64_t) * (rows + 1) + (sizeof(int64_t) + sizeof(double)) * nnz;

    sort_mtx(nnz, mtx, by_row_mtx);

    int64_t *i = SPMV_ALLOC(int64_t, rows + 1);
    // compute row offsets
    i[0] = 0;
    for (int l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) l++;
        i[k + 1] = l;
    }
    // parallel initialization
    csr->i[0] = 0;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < csr->rows; k++) {
        csr->i[k + 1] = i[k + 1];
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            csr->j[l] = mtx[l].j;
            csr->A[l] = mtx[l].a;
        }
    }
    free(i);

    return csr;
}
