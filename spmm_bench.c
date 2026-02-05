#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "spmv.h"

int main(int argc, char *argv[])
{
    int max_threads = omp_get_max_threads();

    if (argc != 3) {
        printf("Usage: spmm_bench <mtx file> <rank>\n");
        return 1;
    }

    int rows, columns, nnz;
    struct mtx *coo;
    load_mtx(argv[1], &rows, &columns, &nnz, &coo);
    int rank = atoi(argv[2]);

    double *x = SPMV_ALLOC(double, columns * rank);
    double *y = SPMV_ALLOC(double, rows * rank);
    for (long i = 0; i < columns * rank; i++) x[i] = 1.0;
    for (long i = 0; i < rows * rank; i++) y[i] = 0.0;

    struct csr *csr = create_csr(rows, columns, nnz, 1, coo);

    // timing
    double t1 = get_time();
    for (int j = 0; j < rank; j++) {
        mult_csr(csr, x + columns * j, y + rows * j);
    }
    double t2 = get_time();
    mult_mv_csr(csr, rank, x, y);
    double t3 = get_time();
    double time_spmv = t2 - t1;
    double time_spmm = t3 - t2;

    // check result
    double *y_ref = SPMV_ALLOC(double, rows);
    mult_csr(csr, x, y_ref);
    double res = 0.0;
    for (int j = 0; j < rank; j++) {
        for (int i = 0; i < rows; i++) {
            double d = fabs(y[i * rank + j] - y_ref[i]);
            // if (d > 1e-5) printf("diff %d %e %e\n", i, y[i], y_ref[i]);
            if (d > res) res = d;
        }
    }

    char *filename = basename(argv[1]);
    char *extension = strrchr(filename, '.');
    if (extension) *extension = 0;
    printf("%s %i %i %i %i %i %e %e %e %e %e\n",
            filename, rank, max_threads,
            rows, columns, nnz,
            2.0 * nnz * rank / time_spmv,
            2.0 * nnz * rank / time_spmm,
            time_spmv, time_spmm, res);
    return 0;
}
