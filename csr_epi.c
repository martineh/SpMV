#include "spmv.h"

void mult_csr_epi(struct csr_epi *csr, double *x, double *y)
{
#ifdef EPI
#if 0
    int gvlmax = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);
    __epi_1xf64 z = __builtin_epi_vsetfirst_1xf64(0, gvlmax);
    for (int k = 0; k < csr->rows; k++) {
        int l = csr->i[k + 1] - csr->i[k];
        if (l <= gvlmax) { // short row
            int gvl = __builtin_epi_vsetvl(l, __epi_e64, __epi_m1);
            __epi_1xf64 a = __builtin_epi_vload_1xf64(csr->A + csr->i[k], gvl);
            __epi_1xi64 j = __builtin_epi_vload_1xi64(csr->j + csr->i[k], gvl);
            __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
            __epi_1xf64 s = __builtin_epi_vfmul_1xf64(a, v, gvl);
            __epi_1xf64 t = __builtin_epi_vfredsum_1xf64(s, z, gvl);
            y[k] = __builtin_epi_vgetfirst_1xf64(t);
        } else {
            int r = l % gvlmax;
            __epi_1xf64 s = __builtin_epi_vbroadcast_1xf64(0, gvlmax);
            if (r > 0) { // row size not multiple of vlmax
                // This should the last iteration of the loop,
                // but doesn't work. Bug in EPAC?
                int gvl = __builtin_epi_vsetvl(r, __epi_e64, __epi_m1);
                __epi_1xf64 a = __builtin_epi_vload_1xf64(csr->A + csr->i[k], gvl);
                __epi_1xi64 j = __builtin_epi_vload_1xi64(csr->j + csr->i[k], gvl);
                __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
                s = __builtin_epi_vfmacc_1xf64(s, a, v, gvl);
            }
            for (int l = csr->i[k] + r; l < csr->i[k + 1]; l += gvlmax) {
                __epi_1xf64 a = __builtin_epi_vload_1xf64(csr->A + l, gvlmax);
                __epi_1xi64 j = __builtin_epi_vload_1xi64(csr->j + l, gvlmax);
                __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvlmax);
                s = __builtin_epi_vfmacc_1xf64(s, a, v, gvlmax);
            }
            __epi_1xf64 t = __builtin_epi_vfredsum_1xf64(s, z, gvlmax);
            y[k] =  __builtin_epi_vgetfirst_1xf64(t);
        }
    }
#else
    // set y to zero
    int gvlmax = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);
    __epi_1xf64 z = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
    for (int i = 0; i < csr->columns; ) {
        int gvl = __builtin_epi_vsetvl(csr->columns - i, __epi_e64, __epi_m1);
        __builtin_epi_vstore_1xf64(y + i, z, gvl);
        i += gvl;
    }
    int last_element = csr->i[csr->rows];
    int n, m = csr->i[1] - csr->i[0];
    for (int l = 0, k = 0; l < last_element;) {
        // read and multiply the maximun number of elements
        int gvl = __builtin_epi_vsetvl(last_element - l, __epi_e64, __epi_m1);
        __epi_1xf64 a = __builtin_epi_vload_1xf64(csr->A + csr->i[k + 1] - m, gvl);
        __epi_1xi64 j = __builtin_epi_vload_1xi64(csr->j + csr->i[k + 1] - m, gvl);
        __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
        __epi_1xf64 s = __builtin_epi_vfmul_1xf64(a, v, gvl);
        l += gvl;

        while (gvl > 0) {
            // reduce and update y[k]
            if (m > gvl) {
                n = __builtin_epi_vsetvl(gvl, __epi_e64, __epi_m1);
            } else {
                n = __builtin_epi_vsetvl(m, __epi_e64, __epi_m1);
            }
            // printf("k %d n %d m %d gvl %d\n", k, n, m, gvl);
            __epi_1xf64 t = __builtin_epi_vfredsum_1xf64(s, z, n);
            y[k] += __builtin_epi_vgetfirst_1xf64(t);
            // update m and k
            m -= n;
            if (m <= 0) {
                // next row
                k = k + 1;
                m = csr->i[k + 1] - csr->i[k];
            }
            // slide if necessary
            if (n < gvl) {
                s = __builtin_epi_vslidedown_1xf64(s, n, gvl);
            }
            gvl -= n;
        }
    }
#endif
#else
    for (int k = 0; k < csr->rows; k++) {
        double s = 0.0;
        #pragma omp simd
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            s += x[csr->j[l]] * csr->A[l];
        }
        y[k] = s;
    }
#endif
}

struct csr_epi *create_csr_epi(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct csr_epi *csr = SPMV_ALLOC(struct csr_epi, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;
    csr->i = SPMV_ALLOC(long, rows + 1);
    csr->j = SPMV_ALLOC(long, nnz);
    csr->A = SPMV_ALLOC(double, nnz);
    csr->memusage = sizeof(int) * (rows + 1) + (sizeof(long) + sizeof(double)) * nnz;

    sort_mtx(nnz, mtx, by_row_mtx);

    csr->i[0] = 0;
    for (int l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) {
            #ifdef EPI
            csr->j[l] = mtx[l].j * 8;
            #else
            csr->j[l] = mtx[l].j;
            #endif
            csr->A[l] = mtx[l].a;
            l++;
        }
        csr->i[k + 1] = l;
    }

    return csr;
}
