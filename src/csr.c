#include "spmv.h"
#include <sys/prctl.h>

void mult_csr_base(struct csr *csr, double *x, double *y)
{
    for (int k = 0; k < csr->rows; k++) {
        double s = 0.0;
        for (int l = csr->i[k]; l < csr->i[k + 1]; l++) {
            s += x[csr->j[l]] * csr->A[l];
        }
        y[k] = s;
    }
}


#if defined(NEON)

#include <arm_neon.h>

void mult_csr(struct csr *csr, double *x, double *y)
{
    int k;
    for (int row = 0; row < csr->rows; ++row) {
        float64x2_t sum = vdupq_n_f64(0.0f);
        for (k = csr->i[row]; k < csr->i[row + 1] - 1; k+=2) {
            float64x2_t val        = vld1q_f64(&csr->A[k]);
	    //Manual gather
            float64x2_t b_vals_vec = {x[csr->j[k]], x[csr->j[k+1]]};
            sum                    = vfmaq_f64(sum, val, b_vals_vec);
        }
        y[row] = sum[0] + sum[1];
        if (k < csr->i[row + 1]) {
	  y[row] += csr->A[k] * x[csr->j[k]];
	}
    }
}

#elif defined(SVE_128) | defined(SVE_256)

#include <arm_sve.h>


void mult_csr(struct csr *csr, double *x, double *y)
{
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
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
}

#elif AVX2

#include <immintrin.h>

#define CSR_AVX2 4

void mult_csr(struct csr *csr, double *x, double *y)
{
    int l;
    for (int k = 0; k < csr->rows; k++) {
        int *j = csr->j + csr->i[k];
        double *A = csr->A + csr->i[k];
        __m256d s = _mm256_setzero_pd();
        for (l = csr->i[k]; l < csr->i[k + 1] - (CSR_AVX2 - 1); l += CSR_AVX2) {
            __m128i vj = _mm_stream_load_si128((__m128i *)j);
            __m256d vx = _mm256_i32gather_pd(x, vj, 8);
            __m256d va = (__m256d)_mm256_stream_load_si256((__m256i *)A);
            s = _mm256_fmadd_pd(vx, va, s);
            A += CSR_AVX2;
            j += CSR_AVX2;
        }

        //y[k] = s[0] + s[1] + s[2] + s[3];
        __m128d a = _mm256_castpd256_pd128(s);
        __m128d b = _mm256_extractf128_pd(s, 1);
        __m128d c = _mm_add_pd(a, b);
        __m128d d = _mm_add_sd(c, _mm_unpackhi_pd(c, c));
        y[k] = _mm_cvtsd_f64(d);

        for (; l < csr->i[k + 1]; l++) y[k] += x[*j++] * *A++;

    }
}

#endif


struct csr *create_csr_pad(int rows, int columns, int nnz, struct mtx *mtx)
{
    return create_csr(rows, columns, nnz, _BLOCK, mtx);
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

void free_csr(struct csr *csr) {
    SPMV_FREE(csr->i);
    SPMV_FREE(csr->j);
    SPMV_FREE(csr->A);
    SPMV_FREE(csr);
}
