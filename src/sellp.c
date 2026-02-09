#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "sellp.h"
#include "spmv.h"
    
struct mtx_coo {int i; int j; double a;};

#if defined(RVV1_M2_256)

#include <riscv_vector.h>

void mult_sellp(struct sellp *sellp, double *x, double *y)
{
  vfloat64m2_t va, vx, vprod;
  vuint32m1_t v_idx;
  uint32_t *col_idx  = (uint32_t *)sellp->j; 
  size_t vl;
  vl = __riscv_vsetvl_e64m2(_BLOCK);
  for (int b = 0; b < sellp->num_blocks; b++) {
    vprod = __riscv_vfmv_v_f_f64m2(0.0, vl);
    for (int l = sellp->block_ptr[b]; l < sellp->block_ptr[b + 1]; l += _BLOCK) {
      va    = __riscv_vle64_v_f64m2(&sellp->A[l],  vl);

      v_idx = __riscv_vle32_v_u32m1(&col_idx[l], vl);
      v_idx = __riscv_vsll_vx_u32m1(v_idx, 3, vl);
      vx = __riscv_vloxei32_v_f64m2(x, v_idx, vl);

      vprod = __riscv_vfmacc_vv_f64m2(vprod, va, vx, vl);
    }
    __riscv_vse64_v_f64m2(&y[b * _BLOCK], vprod, vl);
  }
}



#elif defined(NEON)

#include <arm_neon.h>

void mult_sellp(struct sellp *sellp, double *x, double *y) {
    for (int b = 0; b < sellp->num_blocks; b++) {
	int bstart      = sellp->block_ptr[b];
	int bend        = sellp->block_ptr[b+1];
        float64x2_t sum = vdupq_n_f64(0.0f);
        for (int l = bstart; l < bend; l += _BLOCK) {
            float64x2_t val        = vld1q_f64(&sellp->A[l]);
            float64x2_t b_vals_vec = {x[sellp->j[l]], x[sellp->j[l+1]]};
            sum                    = vfmaq_f64(sum, val, b_vals_vec);
	}
	vst1q_f64(&y[b * _BLOCK], sum);
    }
}

#elif defined(SVE_128) || defined(SVE_256)

#include <arm_sve.h>

#if _BLOCK == 8

//Base implementation
void mult_sellp_b8(struct sellp *sellp, double *x, double *y) {
    const svbool_t pg = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    svfloat64_t va, vx, vprod;
    svint64_t vidx;
    size_t vl;
    for (int b = 0; b < sellp->num_blocks; b++) {
        svfloat64_t sum0 = zeros;
        svfloat64_t sum1 = zeros;
	
	int bstart = sellp->block_ptr[b];
	int bend   = sellp->block_ptr[b+1];

	int nnz = bend - bstart;

	__builtin_prefetch((const char *)&sellp->A[bstart + 8], 0, 3);
	__builtin_prefetch((const char *)&sellp->j[bstart + 8], 0, 3);

        for (int l = bstart; l < bend; l += _BLOCK) {

            svfloat64_t val0   = svld1(pg, &sellp->A[l]);
            svfloat64_t val1   = svld1(pg, &sellp->A[l+4]);
    
            svint64_t col0     = svld1sw_s64(pg, &sellp->j[l]);
            svint64_t col1     = svld1sw_s64(pg, &sellp->j[l+4]);
    
            svfloat64_t bvals0 = svld1_gather_index(pg, x, col0);
            svfloat64_t bvals1 = svld1_gather_index(pg, x, col1);
    
            sum0               = svmla_m(pg, sum0, val0, bvals0);
            sum1               = svmla_m(pg, sum1, val1, bvals1);
        }
        svst1(pg, &y[b * _BLOCK],   sum0);
        svst1(pg, &y[b * _BLOCK+4], sum1);
	
    }
}

void mult_sellp(struct sellp *sellp, double *x, double *y) {
    const svbool_t pg = svptrue_b64();
    //const svfloat64_t zeros = svdup_n_f64(0);
    svfloat64_t va, vx, vprod, sum0, sum1;
    svint64_t vidx;
    size_t vl;
    for (int b = 0; b < sellp->num_blocks; b++) {
        //svfloat64_t sum0 = zeros;
        //svfloat64_t sum1 = zeros;
	
	int bstart = sellp->block_ptr[b];
	int bend   = sellp->block_ptr[b+1];

	int nblock = bend - bstart;

	if (nblock <= 0) {
            y[b] = 0;	
	} else if (0 && nblock == 56) {
 
	    svfloat64_t S0, S1, 
                        A0, A1, A2, A3, A4, A5, A6, A7,
                        X0, X1, X2, X3, X4, X5, X6, X7;

            svint64_t   J0, J1, J2, J3, J4, J5, J6, J7;

            A0 = svld1(pg, &sellp->A[bstart+0]);
            A1 = svld1(pg, &sellp->A[bstart+4]);  //Iter+0
            A2 = svld1(pg, &sellp->A[bstart+8]);
            A3 = svld1(pg, &sellp->A[bstart+12]); //Iter+1
            A4 = svld1(pg, &sellp->A[bstart+16]);
	    A5 = svld1(pg, &sellp->A[bstart+20]); //Iter+2
            A6 = svld1(pg, &sellp->A[bstart+24]);
            A7 = svld1(pg, &sellp->A[bstart+28]); //Iter+3

            J0 = svld1sw_s64(pg, &sellp->j[bstart+0]);
            J1 = svld1sw_s64(pg, &sellp->j[bstart+4]);
            J2 = svld1sw_s64(pg, &sellp->j[bstart+8]);
            J3 = svld1sw_s64(pg, &sellp->j[bstart+12]);
            J4 = svld1sw_s64(pg, &sellp->j[bstart+16]);
            J5 = svld1sw_s64(pg, &sellp->j[bstart+20]);
            J6 = svld1sw_s64(pg, &sellp->j[bstart+24]);
            J7 = svld1sw_s64(pg, &sellp->j[bstart+28]);

            X0 = svld1_gather_index(pg, x, J0);
            X1 = svld1_gather_index(pg, x, J1);
            X2 = svld1_gather_index(pg, x, J2);
            X3 = svld1_gather_index(pg, x, J3);
            X4 = svld1_gather_index(pg, x, J4);
            X5 = svld1_gather_index(pg, x, J5);
            X6 = svld1_gather_index(pg, x, J6);
            X7 = svld1_gather_index(pg, x, J7);

            S0 = svmul_m(pg, A0, X0);
            S1 = svmul_m(pg, A1, X1);
            S0 = svmla_m(pg, S0, A2, X2);
            S1 = svmla_m(pg, S1, A3, X3);
            S0 = svmla_m(pg, S0, A4, X4);
            S1 = svmla_m(pg, S1, A5, X5);
            S0 = svmla_m(pg, S0, A6, X6);
            S1 = svmla_m(pg, S1, A7, X7);

            A0 = svld1(pg, &sellp->A[bstart+32]);
            A1 = svld1(pg, &sellp->A[bstart+36]);  //Iter+0
            A2 = svld1(pg, &sellp->A[bstart+40]);
            A3 = svld1(pg, &sellp->A[bstart+44]); //Iter+1
            A4 = svld1(pg, &sellp->A[bstart+48]);
	    A5 = svld1(pg, &sellp->A[bstart+52]); //Iter+2
            
	    J0 = svld1sw_s64(pg, &sellp->j[bstart+32]);
            J1 = svld1sw_s64(pg, &sellp->j[bstart+36]);
            J2 = svld1sw_s64(pg, &sellp->j[bstart+40]);
            J3 = svld1sw_s64(pg, &sellp->j[bstart+44]);
            J4 = svld1sw_s64(pg, &sellp->j[bstart+48]);
            J5 = svld1sw_s64(pg, &sellp->j[bstart+52]);
            
	    X0 = svld1_gather_index(pg, x, J0);
            X1 = svld1_gather_index(pg, x, J1);
            X2 = svld1_gather_index(pg, x, J2);
            X3 = svld1_gather_index(pg, x, J3);
            X4 = svld1_gather_index(pg, x, J4);
            X5 = svld1_gather_index(pg, x, J5);
            
	    S0 = svmla_m(pg, S0, A0, X0);
            S1 = svmla_m(pg, S1, A1, X1);
            S0 = svmla_m(pg, S0, A2, X2);
            S1 = svmla_m(pg, S1, A3, X3);
            S0 = svmla_m(pg, S0, A4, X4);
            S1 = svmla_m(pg, S1, A5, X5);
        
            svst1(pg, &y[b * _BLOCK],   S0);
            svst1(pg, &y[b * _BLOCK+4], S1);
	} else {
	    __builtin_prefetch((const char *)&sellp->A[bstart + 8], 0, 3);
	    __builtin_prefetch((const char *)&sellp->j[bstart + 8], 0, 3);
    
            svfloat64_t val0   = svld1(pg, &sellp->A[bstart]);
            svfloat64_t val1   = svld1(pg, &sellp->A[bstart+4]);
        
            svint64_t col0     = svld1sw_s64(pg, &sellp->j[bstart]);
            svint64_t col1     = svld1sw_s64(pg, &sellp->j[bstart+4]);
        
            svfloat64_t bvals0 = svld1_gather_index(pg, x, col0);
            svfloat64_t bvals1 = svld1_gather_index(pg, x, col1);
        
            sum0               = svmul_f64_m(pg, val0, bvals0);
            sum1               = svmul_f64_m(pg, val1, bvals1);

            for (int l = bstart + _BLOCK; l < bend; l += _BLOCK) {
    
                svfloat64_t val0   = svld1(pg, &sellp->A[l]);
                svfloat64_t val1   = svld1(pg, &sellp->A[l+4]);
        
                svint64_t col0     = svld1sw_s64(pg, &sellp->j[l]);
                svint64_t col1     = svld1sw_s64(pg, &sellp->j[l+4]);
        
                svfloat64_t bvals0 = svld1_gather_index(pg, x, col0);
                svfloat64_t bvals1 = svld1_gather_index(pg, x, col1);
        
                sum0               = svmla_m(pg, sum0, val0, bvals0);
                sum1               = svmla_m(pg, sum1, val1, bvals1);
            }
            svst1(pg, &y[b * _BLOCK],   sum0);
            svst1(pg, &y[b * _BLOCK+4], sum1);
	}
	
    }
}

#elif _BLOCK == 16

void mult_sellp(struct sellp *sellp, double *x, double *y) {
    const svbool_t pg = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    svfloat64_t va, vx, vprod, sum0, sum1;
    svint64_t vidx;
    size_t vl;
    for (int b = 0; b < sellp->num_blocks; b++) {
        svfloat64_t sum0 = zeros;
        svfloat64_t sum1 = zeros;
        svfloat64_t sum2 = zeros;
        svfloat64_t sum3 = zeros;
	
	int bstart = sellp->block_ptr[b];
	int bend   = sellp->block_ptr[b+1];

	__builtin_prefetch((const char *)&sellp->A[bstart + 8], 0, 3);
	__builtin_prefetch((const char *)&sellp->j[bstart + 8], 0, 3);
    
        for (int l = bstart; l < bend; l += _BLOCK) {
    
            svfloat64_t val0   = svld1(pg, &sellp->A[l]);
            svfloat64_t val1   = svld1(pg, &sellp->A[l+4]);
            svfloat64_t val2   = svld1(pg, &sellp->A[l+8]);
            svfloat64_t val3   = svld1(pg, &sellp->A[l+12]);
    
            svint64_t col0     = svld1sw_s64(pg, &sellp->j[l]);
            svint64_t col1     = svld1sw_s64(pg, &sellp->j[l+4]);
            svint64_t col2     = svld1sw_s64(pg, &sellp->j[l+8]);
            svint64_t col3     = svld1sw_s64(pg, &sellp->j[l+12]);
    
            svfloat64_t bvals0 = svld1_gather_index(pg, x, col0);
            svfloat64_t bvals1 = svld1_gather_index(pg, x, col1);
            svfloat64_t bvals2 = svld1_gather_index(pg, x, col2);
            svfloat64_t bvals3 = svld1_gather_index(pg, x, col3);
    
            sum0               = svmla_m(pg, sum0, val0, bvals0);
            sum1               = svmla_m(pg, sum1, val1, bvals1);
            sum2               = svmla_m(pg, sum2, val2, bvals2);
            sum3               = svmla_m(pg, sum3, val3, bvals3);
        }

        svst1(pg, &y[b * _BLOCK+0],  sum0);
        svst1(pg, &y[b * _BLOCK+4],  sum1);
        svst1(pg, &y[b * _BLOCK+8],  sum2);
        svst1(pg, &y[b * _BLOCK+12], sum3);
    }
	
}

#else

void mult_sellp(struct sellp *sellp, double *x, double *y) {
    const svbool_t pg = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    for (int b = 0; b < sellp->num_blocks; b++) {
        svfloat64_t sum = zeros;
        for (int l = sellp->block_ptr[b]; l < sellp->block_ptr[b + 1]; l += _BLOCK) {
            svfloat64_t val        = svld1(pg, &sellp->A[l]);
            svint64_t col          = svld1sw_s64(pg, &sellp->j[l]);
            svfloat64_t b_vals_vec = svld1_gather_index(pg, x, col);
            sum                    = svmla_m(pg, sum, val, b_vals_vec);
        }
        svst1(pg, &y[b * _BLOCK], sum);
    }
}

#endif

#elif AVX2

#include <immintrin.h>

void mult_sellp(struct sellp *sellp, double *x, double *y) {
    for (int b = 0; b < sellp->num_blocks; b++) {
        __m256d s = _mm256_setzero_pd();
        for (int l = sellp->block_ptr[b]; l < sellp->block_ptr[b + 1]; l += _BLOCK) {
            __m256d va = _mm256_castsi256_pd(_mm256_stream_load_si256((__m256i *)&sellp->A[l]));
            __m128i vj = _mm_stream_load_si128((__m128i *)&sellp->j[l]);
            __m256d vx = _mm256_i32gather_pd(x, vj, 8);
            s = _mm256_fmadd_pd(vx, va, s);
        }
        _mm256_stream_pd(&y[b * _BLOCK], s);
    }
    
}

/*
void mult_sellp(struct sellp *ell, double *x, double *y)
{
    int r = ell->rows - ell->num_blocks * _BLOCK; // remainder rows
    for (int b = 0; b < ell->num_blocks; b++) {
        double t[_BLOCK];
        for (int s = 0; s < _BLOCK; s++) t[s] = 0;
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += _BLOCK) {
            for (int s = 0; s < _BLOCK; s++) {
                t[s] += x[ell->j[l + s]] * ell->A[l + s];
            }
        }
        if (b == ell->num_blocks - 1 && r > 0) {
            for (int s = 0; s < r; s++) y[b * _BLOCK + s] = t[s];
        } else {
            for (int s = 0; s < _BLOCK; s++) y[b * _BLOCK + s] = t[s];
        }
    }
}
*/

#endif

int by_row(const struct mtx_coo *a, const struct mtx_coo *b) {
    if (a->i < b->i) return -1;
    if (a->i > b->i) return 1;
    if (a->j < b->j) return -1;
    if (a->j > b->j) return 1;
    return 0;
}

struct sellp *create_sellp(int rows, int columns, int nnz, const int *row_ptr_csr, const int *col_idx, const double *A_csr) {
    struct sellp *sellp = (struct sellp *)aligned_alloc(ALIGN_BYTES, sizeof(*sellp));
    if (!sellp) { perror("aligned_alloc sellp"); exit(1); }

    sellp->rows = rows;
    sellp->columns = columns;
    sellp->nnz = nnz;

    // Número de bloques
    int num_blocks = rows / _BLOCK;
    if (rows % _BLOCK) num_blocks++;
    sellp->num_blocks = num_blocks;

    // block_ptr alineado
    if (posix_memalign((void**)&sellp->block_ptr, ALIGN_BYTES, sizeof(int) * (num_blocks + 1))) {
        perror("posix_memalign block_ptr"); exit(1);
    }
    sellp->memusage = sizeof(int) * (num_blocks + 1);

    // Crear COO a partir de CSR

    struct mtx_coo *mtx = (struct mtx_coo *)aligned_alloc(ALIGN_BYTES, sizeof(struct mtx_coo) * nnz);
    if (!mtx) { perror("aligned_alloc mtx"); exit(1); }

    int pos = 0;
    for (int r = 0; r < rows; r++) {
        for (int k = row_ptr_csr[r]; k < row_ptr_csr[r+1]; k++) {
            mtx[pos].i = r;
            mtx[pos].j = col_idx[k];
            mtx[pos].a = A_csr[k];
            pos++;
        }
    }


    // Ordenar por filas, columnas
    bool sorted = true;
    int ii = 1;
    while (sorted && ii < nnz) {
        if (by_row(&mtx[ii-1], &mtx[ii]) == 1) sorted = false;
        ii++;
    }
    if (!sorted) {
        qsort(mtx, nnz, sizeof(struct mtx_coo),
              (int (*)(const void *, const void *))by_row);
    }

    // Construir row_ptr temporal
    int *row_ptr_temp;
    if (posix_memalign((void**)&row_ptr_temp, ALIGN_BYTES, sizeof(int) * (num_blocks * _BLOCK + 1))) {
        perror("posix_memalign row_ptr"); exit(1);
    }
    row_ptr_temp[0] = 0;
    for (int l = 0, k = 0; k < num_blocks * _BLOCK; k++) {
        while (l < nnz && mtx[l].i == k) l++;
        row_ptr_temp[k + 1] = l;
    }

    // Calcular block_ptr con padding
    sellp->block_ptr[0] = 0;
    for (int p = 0, b = 0; b < num_blocks; b++) {
        int max_length = 0;
        for (int s = 0; s < _BLOCK; s++) {
            int row_len = row_ptr_temp[b * _BLOCK + s + 1] -
                          row_ptr_temp[b * _BLOCK + s];
            if (row_len > max_length) max_length = row_len;
        }
        p += max_length;
        sellp->block_ptr[b + 1] = p * _BLOCK;
    }

    int l_total = sellp->block_ptr[num_blocks];

    // Reservar j y A alineados
    if (posix_memalign((void**)&sellp->j, ALIGN_BYTES, sizeof(int) * l_total) ||
        posix_memalign((void**)&sellp->A, ALIGN_BYTES, sizeof(double) * l_total)) {
        perror("posix_memalign j/A"); exit(1);
    }
    sellp->memusage += (sizeof(int) + sizeof(double)) * l_total;

    // Llenar la matriz SELL-P
    for (int b = 0; b < num_blocks; b++) {
        int row_len[_BLOCK];
        for (int s = 0; s < _BLOCK; s++) {
            row_len[s] = row_ptr_temp[b * _BLOCK + s + 1] -
                         row_ptr_temp[b * _BLOCK + s];
        }

        for (int l = sellp->block_ptr[b]; l < sellp->block_ptr[b+1]; l += _BLOCK) {
            for (int s = 0; s < _BLOCK; s++) {
                if (row_len[s] == 0) { // padding
                    sellp->j[l + s] = 0;
                    sellp->A[l + s] = 0.0;
                } else {
                    sellp->j[l + s] = mtx[row_ptr_temp[b * _BLOCK + s]].j;
                    sellp->A[l + s] = mtx[row_ptr_temp[b * _BLOCK + s]].a;
                    row_ptr_temp[b * _BLOCK + s]++;
                    row_len[s]--;
                }
            }
        }
    }

    free(row_ptr_temp);
    free(mtx);
    return sellp;
}


void free_sellp(struct sellp *sellp) {
    free(sellp->j);
    free(sellp->A);
    free(sellp->block_ptr);
}
