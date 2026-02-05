#include "spmv.h"
#include <sys/prctl.h>


#include "arm_sve.h"

#define ALIGNMENT 1

//Scalar version
void mult_csr_base(struct csr *csr, double *x, double *y)
{
    for (int k = 0; k < csr->rows; k++) {
        double s = 0.0;
        int *j = csr->j + csr->i[k];
        double *A = csr->A + csr->i[k];
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            s += x[*j++] * *A++;
        }
        y[k] = s;
    }
}



//SVE2 - base Implementation
void mult_csr(struct csr *csr, double *x, double *y)
{
    //size_t chunks[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    //size_t total_nnz = 0;
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    for (int row = 0; row < csr->rows; ++row) {
        svfloat64_t sum = zeros;
	
	//-------------------------------------------------------
	//int nnz = csr->i[row + 1] - csr->i[row];
	//if (nnz <= 9) chunks[nnz]++;
	//else          chunks[9]++;
	//total_nnz += nnz;
	//-------------------------------------------------------
	
        for (int k = csr->i[row]; k < csr->i[row + 1]; k+=svcntd()) {
            svbool_t pg            = svwhilelt_b64(k, csr->i[row + 1]);
            svfloat64_t val        = svld1(pg, &csr->A[k]);
            svint64_t col          = svld1sw_s64(pg, &csr->j[k]);
            svfloat64_t b_vals_vec = svld1_gather_index(pg, x, col);
            sum                    = svmla_m(pg, sum, val, b_vals_vec);
        }
        y[row] = svaddv(true_vec, sum);
    }

    //printf("%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu; (%zu)\n", chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5], chunks[6], chunks[7], chunks[8], chunks[9], total_nnz);
}





void mult_csr_s(struct csr *csr, double *x, double *y)
{

    //int new_vl_in_bytes = 16;
    //prctl(PR_SVE_SET_VL, new_vl_in_bytes);

    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    svfloat64_t b_vals_vec;

    for (int row = 0; row < csr->rows; ++row) {
            svfloat64_t sum = zeros;
            for (int k = csr->i[row]; k < csr->i[row + 1]; k+=svcntd()) {
                svbool_t pg            = svwhilelt_b64(k, csr->i[row + 1]);
                svfloat64_t val        = svld1(pg, &csr->A[k]);

                svint64_t col          = svld1sw_s64(pg, &csr->j[k]);
		svfloat64_t b_vals_vec = svld1_gather_index(pg, x, col);

                sum                    = svmla_m(pg, sum, val, b_vals_vec);
            }
            y[row] = svaddv(true_vec, sum);
	}
    //}

    //new_vl_in_bytes = 16;
    //prctl(PR_SVE_SET_VL, new_vl_in_bytes);

}



void mult_mv_csr(struct csr *csr, int n, double *x, double *y)
{
#ifdef __AVX2__
    if (n == 4) {
        #pragma omp parallel for
        for (int k = 0; k < csr->rows; k++) {
            register __m256d vy = _mm256_setzero_pd();
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                register __m256d vx = _mm256_load_pd(x + csr->j[l] * n);
                register __m256d va = _mm256_broadcast_sd(csr->A + l);
                vy = _mm256_fmadd_pd(vx, va, vy);
            }
            _mm256_store_pd(y + k * n, vy);
        }
    } else if (n == 8) {
        #pragma omp parallel for
        for (int k = 0; k < csr->rows; k++) {
            register __m256d vy0 = _mm256_setzero_pd();
            register __m256d vy1 = _mm256_setzero_pd();
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                register __m256d va = _mm256_broadcast_sd(csr->A + l);
                register double *px = x + csr->j[l] * n;
                vy0 = _mm256_fmadd_pd(_mm256_load_pd(px), va, vy0);
                vy1 = _mm256_fmadd_pd(_mm256_load_pd(px + 4), va, vy1);
            }
            double *py = y + k * n;
            _mm256_store_pd(py, vy0);
            _mm256_store_pd(py + 4, vy1);
        }
    } else if (n == 16) {
        #pragma omp parallel for
        for (int k = 0; k < csr->rows; k++) {
            register __m256d vy0 = _mm256_setzero_pd();
            register __m256d vy1 = _mm256_setzero_pd();
            register __m256d vy2 = _mm256_setzero_pd();
            register __m256d vy3 = _mm256_setzero_pd();
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                register __m256d va = _mm256_broadcast_sd(csr->A + l);
                register double *px = x + csr->j[l] * n;
                vy0 = _mm256_fmadd_pd(_mm256_load_pd(px), va, vy0);
                vy1 = _mm256_fmadd_pd(_mm256_load_pd(px + 4), va, vy1);
                vy2 = _mm256_fmadd_pd(_mm256_load_pd(px + 8), va, vy2);
                vy3 = _mm256_fmadd_pd(_mm256_load_pd(px + 12), va, vy3);
            }
            double *py = y + k * n;
            _mm256_store_pd(py, vy0);
            _mm256_store_pd(py + 4, vy1);
            _mm256_store_pd(py + 8, vy2);
            _mm256_store_pd(py + 12, vy3);
        }
    } else if (n == 32) {
        #pragma omp parallel for
        for (int k = 0; k < csr->rows; k++) {
            register __m256d vy0 = _mm256_setzero_pd();
            register __m256d vy1 = _mm256_setzero_pd();
            register __m256d vy2 = _mm256_setzero_pd();
            register __m256d vy3 = _mm256_setzero_pd();
            register __m256d vy4 = _mm256_setzero_pd();
            register __m256d vy5 = _mm256_setzero_pd();
            register __m256d vy6 = _mm256_setzero_pd();
            register __m256d vy7 = _mm256_setzero_pd();
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                register __m256d va = _mm256_broadcast_sd(csr->A + l);
                register double *px = x + csr->j[l] * n;
                vy0 = _mm256_fmadd_pd(_mm256_load_pd(px), va, vy0);
                vy1 = _mm256_fmadd_pd(_mm256_load_pd(px + 4), va, vy1);
                vy2 = _mm256_fmadd_pd(_mm256_load_pd(px + 8), va, vy2);
                vy3 = _mm256_fmadd_pd(_mm256_load_pd(px + 12), va, vy3);
                vy4 = _mm256_fmadd_pd(_mm256_load_pd(px + 16), va, vy4);
                vy5 = _mm256_fmadd_pd(_mm256_load_pd(px + 20), va, vy5);
                vy6 = _mm256_fmadd_pd(_mm256_load_pd(px + 24), va, vy6);
                vy7 = _mm256_fmadd_pd(_mm256_load_pd(px + 28), va, vy7);
            }
            double *py = y + k * n;
            _mm256_store_pd(py, vy0);
            _mm256_store_pd(py + 4, vy1);
            _mm256_store_pd(py + 8, vy2);
            _mm256_store_pd(py + 12, vy3);
            _mm256_store_pd(py + 16, vy4);
            _mm256_store_pd(py + 20, vy5);
            _mm256_store_pd(py + 24, vy6);
            _mm256_store_pd(py + 28, vy7);
        }
    } else
#endif
    {
        #pragma omp parallel for
        for (int k = 0; k < csr->rows; k++) {
            double *py = y + k * n;
            for (int i = 0; i < n; i++) py[i] = 0;
            for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
                double *px = x + csr->j[l] * n;
                double A = csr->A[l];
                #pragma omp simd
                for (int i = 0; i < n; i++) {
                    py[i] += px[i] * A;
                }
            }
        }
    }
}

struct csr *create_csr_pad(int rows, int columns, int nnz, struct mtx *mtx)
{
    return create_csr(rows, columns, nnz, ALIGNMENT, mtx);
}

struct csr *create_csr(int rows, int columns, int nnz, int alignment, struct mtx *mtx)
{
    struct csr *csr = SPMV_ALLOC(struct csr, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;

    sort_mtx(nnz, mtx, by_row_mtx);

    int p = 0;
    for (int l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) { p++; l++; }
        int e = p % alignment;
        if (e != 0) p += alignment - e;
    }

    csr->i = SPMV_ALLOC(int, rows + 1);
    csr->j = SPMV_ALLOC(int, p);
    csr->A = SPMV_ALLOC(double, p);
    csr->memusage = sizeof(int) * (rows + 1) + (sizeof(int) + sizeof(double)) * p;

    csr->i[0] = 0;
    for (int p = 0, l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) {
            csr->j[p] = mtx[l].j;
            csr->A[p] = mtx[l].a;
            p++; l++;
        }
        if (alignment != 1) {
            int e = p % alignment;
            if (e != 0) {
                for (int i = 0; i < alignment - e; i++) {
                    csr->j[p] = 0;
                    csr->A[p] = 0;
                    p++;
                }
            }
        }
        csr->i[k + 1] = p;
    }

    return csr;
}
