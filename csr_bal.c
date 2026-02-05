#include "spmv.h"

void mult_csr_bal(struct csr_bal *csr_bal, double *x, double *y)
{
    double buffer[csr_bal->threads - 1];
    #pragma omp parallel num_threads(csr_bal->threads)
    {
        int t = omp_get_thread_num();
        struct csr_bal_thread *csr = csr_bal->p + t;
        double s = 0.0;
        int *j = csr->j;
        double *A = csr->A;
        #pragma omp simd
        for (int l = csr->i[0]; l < csr->i[1]; l++) {
            s += x[j[l]] * A[l];
        }
        if (t > 0 && csr->first_row == csr_bal->p[t - 1].last_row) {
            // our first row is shared with the previous thread
            buffer[t - 1] = s;
        } else {
            y[csr->first_row] = s;
        }
        for (int k = 1; k < csr->rows; k++) {
            double s = 0.0;
            #pragma omp simd
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                s += x[j[l]] * A[l];
            }
            y[csr->first_row + k] = s;
        }
    }
    // update partial rows
    for (int t = 1; t < csr_bal->threads; t++) {
        if (csr_bal->p[t].first_row == csr_bal->p[t - 1].last_row) {
            y[csr_bal->p[t].first_row] += buffer[t - 1];
        }
    }
}

static void create_csr_bal_thread(struct csr_bal_thread *csr, int nnz, struct mtx *mtx)
{
    csr->first_row = mtx[0].i; // first row number
    csr->last_row = mtx[nnz - 1].i; // last row number
    // compute number of rows
    csr->rows = 0;
    for (int l = 0; l < nnz; l++) {
        if (l == 0 || mtx[l].i != mtx[l - 1].i) {
            csr->rows++;
        }
    }
    // fill CSR vectors
    csr->i = SPMV_ALLOC(int, csr->rows + 1);
    csr->j = SPMV_ALLOC(int, nnz);
    csr->A = SPMV_ALLOC(double, nnz);
    for (int k = 0, l = 0; l < nnz; l++) {
        if (l == 0 || mtx[l].i != mtx[l - 1].i) {
            csr->i[k] = l;
            k++;
        }
        csr->j[l] = mtx[l].j;
        csr->A[l] = mtx[l].a;
    }
    csr->i[csr->rows] = nnz;
}

struct csr_bal *create_csr_bal(int rows, int columns, int nnz, struct mtx *mtx)
{
    int threads = omp_get_max_threads();
    int struct_size = sizeof(struct csr_bal) + sizeof(struct csr_bal_thread) * threads;
    struct csr_bal *csr_bal = (struct csr_bal *)malloc(struct_size);
    csr_bal->memusage = struct_size;
    csr_bal->rows = rows;
    csr_bal->columns = columns;
    csr_bal->nnz = nnz;
    csr_bal->threads = threads;

    sort_mtx(nnz, mtx, by_row_mtx);

    // parallel initialization
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num();
        int length = (nnz + t) / threads; // number of non-zeros
        int start = 0; // first non-zero
        for (int k = 0; k < t; k++) start += (nnz + k) / threads;
        create_csr_bal_thread(csr_bal->p + t, length, mtx + start);
    }

    for (int t = 0; t < threads; t++) {
        csr_bal->memusage += sizeof(int) * csr_bal->p[t].rows;
    }
    csr_bal->memusage += (sizeof(int) + sizeof(double)) * nnz;
    return csr_bal;
}
