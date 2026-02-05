#include "spmv.h"

// #define GLOBAL_SORT

#if 0 // def __AVX2__

#include <immintrin.h>

#define SIMD_WIDTH 4

void mult_csri(struct csri *csr, double *x, double *y)
{
    #pragma omp parallel for
    for (int b = 0; b < csr->nb; b++) {
        int *j = csr->j + csr->block_ptr[b];
        double *A = csr->A + csr->block_ptr[b];
        register __m256d s = _mm256_setzero_pd();
        register __m256d zero = _mm256_setzero_pd();
        for (int k = csr->block_size[b]; k < csr->block_size[b + 1]; k++) {
            switch (csr->block_width[k]) { // access stride
            case 4:
                register __m128i mask128 = _mm_set1_epi32(-1);
                for (int l = 0; l < csr->block_len[k]; l++) {
                    __m128i vj = _mm_maskload_epi32(j, mask128);
                    j += 4;
                    __m256d vx = _mm256_i32gather_pd(x, vj, 8);
                    __m256d va = _mm256_loadu_pd(A);
                    A += 4;
                    s = _mm256_fmadd_pd(vx, va, s);
                }
                break;
            case 3:
                mask128 = _mm_set_epi32(0, -1, -1, -1);
                register __m256i mask256 = _mm256_set_epi64x(0, -1, -1, -1);
                for (int l = 0; l < csr->block_len[k]; l++) {
                    __m128i vj = _mm_maskload_epi32(j, mask128);
                    j += 3;
                    __m256d vx = _mm256_mask_i32gather_pd(zero, x, vj, (__m256d)mask256, 8);
                    __m256d va = _mm256_maskload_pd(A, mask256);
                    A += 3;
                    s = _mm256_fmadd_pd(vx, va, s);
                }
                break;
            case 1:
                int rest = csr->block_len[k] & 0x4;
                register double t = 0;
                for (int l = 0; l < rest; l++) t += x[*j++] * *A++;
                for (int l = rest; l < csr->block_len[k]; l++) t += x[*j++] * *A++;
                s[0] = t;
                break;
            default:
                for (int l = 0; l < csr->block_len[k]; l++) {
                    #pragma omp simd
                    for (int w = 0; w < csr->block_width[k]; w++) {
                        s[w] += x[*j++] * *A++;
                    }
                }
            }
        }
        int *r = csr->row_id + b * SIMD_WIDTH;
        #pragma omp simd
        for (int w = 0; w < csr->block_width[csr->block_size[b]]; w++) {
            y[*r++] = s[w];
        }
    }
}

#else

#define SIMD_WIDTH 256

void mult_csri(struct csri *csr, double *x, double *y)
{
    #pragma omp parallel for
    for (int b = 0; b < csr->nb; b++) {
        int *j = csr->j + csr->block_ptr[b];
        double *A = csr->A + csr->block_ptr[b];
        double s[SIMD_WIDTH]; // this should be a register
        #pragma omp simd
        for (int w = 0; w < SIMD_WIDTH; w++) s[w] = 0;
        for (int k = csr->block_size[b]; k < csr->block_size[b + 1]; k++) {
            for (int l = 0; l < csr->block_len[k]; l++) {
                #pragma omp simd
                for (int w = 0; w < csr->block_width[k]; w++) {
                    s[w] += x[*j++] * *A++;
                }
            }
        }
        int *r = csr->row_id + b * SIMD_WIDTH;
        #pragma omp simd
        for (int w = 0; w < csr->block_width[csr->block_size[b]]; w++) {
            y[*r++] = s[w];
        }
    }
}

#endif

struct aux_row {
    int id;
    int ptr;
    int len;
};

int sort_aux_row(const void *pa, const void *pb)
{
    const struct aux_row *a = pa;
    const struct aux_row *b = pb;
    if (a->len < b->len) return 1;
    if (a->len > b->len) return -1;
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

struct csri *create_csri(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct csri *csr = SPMV_ALLOC(struct csri, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;
    csr->nb = rows / SIMD_WIDTH + (rows % SIMD_WIDTH != 0 ? 1 : 0);
    csr->block_ptr = SPMV_ALLOC(int, csr->nb);
    csr->block_size = SPMV_ALLOC(int, csr->nb + 1);
    csr->row_id = SPMV_ALLOC(int, rows);
    csr->j = SPMV_ALLOC(int, nnz);
    csr->A = SPMV_ALLOC(double, nnz);
    csr->memusage = sizeof(int) * (2 * csr->nb + 1 + rows) + (sizeof(int) + sizeof(double)) * nnz;

    // sort as for CSR
    sort_mtx(nnz, mtx, by_row_mtx);

    // sort rows by their lenght
    struct aux_row *aux = SPMV_ALLOC(struct aux_row, csr->nb * SIMD_WIDTH);
    for (int l = 0, k = 0; k < rows; k++) {
        aux[k].id = k;
        aux[k].ptr = l;
        while (l < nnz && mtx[l].i == k) l++;
        aux[k].len = l - aux[k].ptr;
    }
    for (int k = rows; k < csr->nb * SIMD_WIDTH; k++) {
        aux[k].id = -1;
        aux[k].ptr = -1;
        aux[k].len = 0;
    }
#ifdef GLOBAL_SORT
    qsort(aux, csr->nb * SIMD_WIDTH, sizeof(struct aux_row), sort_aux_row);
#else
    for (int b = 0; b < csr->nb; b++) {
        qsort(aux + b * SIMD_WIDTH, SIMD_WIDTH, sizeof(struct aux_row), sort_aux_row);
    }
#endif
    for (int k = 0; k < rows; k++) {
            csr->row_id[k] = aux[k].id;
    }

    // compute block pointers to size info
    csr->block_size[0] = 0;
    int q = 0;
    for (int b = 0; b < csr->nb; b++) {
        int aux_len[SIMD_WIDTH];
        for (int k = b * SIMD_WIDTH, w = 0; w < SIMD_WIDTH; k++, w++) {
            aux_len[w] = aux[k].len;
        }
        for (int w = SIMD_WIDTH - 1; w >= 0; w--) {
            if (aux_len[w] > 0) {
                for (int t = 0; t <= w; t++) {
                    aux_len[t] -= aux_len[w];
                }
                q++;
            }
        }
        csr->block_size[b + 1] = q;
    }

    // compute block sizes
    csr->block_width = SPMV_ALLOC(int, q);
    csr->block_len = SPMV_ALLOC(int, q);
    csr->memusage += 2 * sizeof(int) * q;
    for (int q = 0, b = 0; b < csr->nb; b++) {
        int aux_len[SIMD_WIDTH];
        for (int w = 0; w < SIMD_WIDTH; w++) {
            aux_len[w] = aux[b * SIMD_WIDTH + w].len;
        }
        for (int w = SIMD_WIDTH - 1; w >= 0; w--) {
            if (aux_len[w] > 0) {
                csr->block_width[q] = w + 1;
                csr->block_len[q] = aux_len[w];
                for (int t = 0; t <= w; t++) {
                    aux_len[t] -= aux_len[w];
                }
                q++;
            }
        }
    }

    // copy column indexes and values
    for (int p = 0, b = 0; b < csr->nb; b++) {
        csr->block_ptr[b] = p;
        int aux_ptr[SIMD_WIDTH];
        for (int w = 0; w < SIMD_WIDTH; w++) {
            aux_ptr[w] = aux[b * SIMD_WIDTH + w].ptr;
        }
        for (int k = csr->block_size[b]; k < csr->block_size[b + 1]; k++) {
            for (int l = 0; l < csr->block_len[k]; l++) {
                for (int w = 0; w < csr->block_width[k]; w++) {
                    csr->j[p] = mtx[aux_ptr[w]].j;
                    csr->A[p] = mtx[aux_ptr[w]].a;
                    p++;
                    aux_ptr[w]++;
                }
            }
        }
    }

    free(aux);
    return csr;
}
