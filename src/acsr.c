#include <cstdlib>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include <stdio.h>
#include "acsr.h"


//spmv base
void mult_acsr (struct acsr *acsr, double *x, double *y) {
    for (int i = 0; i < acsr->rows; i++) {
        double sum = 0.0;

        int start = acsr->v_rowptr[i];
        int end   = acsr->v_rowptr[i + 1];

        for (int j = start; j < end; j++) {
            double a0 = acsr->v_values[j * 2];
            double a1 = acsr->v_values[j * 2 + 1];
            sum += a0 * x[acsr->v_columns[j]];
            sum += a1 * x[acsr->v_columns[j] + 1];
        }
	y[i] = sum;
    }

}


/*
#include <arm_sve.h>
void mult_acsr (struct acsr *acsr, double *x, double *y) {
    int ret = prctl(PR_SVE_SET_VL, 16);
    if (ret < 0) { perror("prctl"); return; }

    //const svbool_t pg = svptrue_b64();
    const svbool_t pg = svwhilelt_b64(0, 2);
    const svfloat64_t zeros = svdup_n_f64(0);


    for (int i = 0; i < acsr->rows; i++) {
        svfloat64_t sv = zeros;
        int start = acsr->v_rowptr[i];
        int end   = acsr->v_rowptr[i + 1];
        for (int j = start; j < end; j++) {
	    svfloat64_t av0 = svld1_f64(pg, acsr->v_values + j * 2);
	    svfloat64_t xv0 = svld1_f64(pg, x + acsr->v_columns[j]);
	    sv = svmla_m(pg, sv, av0, xv0);
        }
        y[i] = svaddv(pg, sv);
    }

    ret = prctl(PR_SVE_SET_VL, 32);
    if (ret < 0) { perror("prctl"); return; }
}
*/

/* Block for vlen=128, x2 doubles */
struct acsr *create_acsr(int nrows, int ncols, int nnz, double *values, int64_t *columns, int64_t *row_ptr) {

    int capacity = row_ptr[nrows] * 2;

    double *v_values  = (double *)aligned_alloc(32, sizeof(double) * capacity);
    int64_t    *v_columns = (int64_t *)aligned_alloc(32,sizeof(int64_t) * capacity);
    int64_t    *v_rowptr  = (int64_t *)aligned_alloc(32,sizeof(int64_t) * (nrows + 1));

    int vec_count = 0;
    int dist, start, end;
    int i, j;

    for (i = 0; i < nrows; i++) {
        v_rowptr[i] = vec_count;

        start = row_ptr[i];
        end   = row_ptr[i + 1];

	j = start;
	while (j < end) {
            if (j + 1 < end) dist = columns[j + 1] - columns[j];
	    else             dist = 0;
	    if (dist == 1) {
                v_values[2*vec_count + 0] = values[j];
                v_values[2*vec_count + 1] = values[j + 1];
                v_columns[vec_count]      = columns[j];
                j += 2;
            } else {
                v_values[2*vec_count + 0] = values[j];
                v_values[2*vec_count + 1] = 0.0;
                v_columns[vec_count]      = columns[j];
                j += 1;
            }
            vec_count++;
        }
    }
    v_rowptr[nrows] = vec_count;

    struct acsr *acsr = (struct acsr *)aligned_alloc(32, sizeof(*acsr));

    acsr->v_values    = v_values;
    acsr->v_columns   = v_columns;
    acsr->v_rowptr    = v_rowptr;
    acsr->num_vectors = vec_count;
    acsr->rows        = nrows;
    acsr->columns     = ncols;
    acsr->nnz         = nnz;

    return acsr;
}


