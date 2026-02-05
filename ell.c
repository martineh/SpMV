#include "spmv.h"

#include <arm_sve.h>

#define SIMD_WIDTH 4

void mult_ell(struct ell *ell, double *x, double *y)
{
  const svbool_t pg = svptrue_b64();
  const svfloat64_t zeros = svdup_n_f64(0);
  svfloat64_t va, vx, vprod;
  svint64_t vidx;
  size_t vl;
  for (int b = 0; b < ell->num_blocks; b++) {
    svfloat64_t sum = zeros;
    for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH) {
      svfloat64_t val        = svld1(pg, &ell->A[l]);
      svint64_t col          = svld1_s64(pg, &ell->j[l]);
      svfloat64_t b_vals_vec = svld1_gather_index(pg, x, col);
      sum                    = svmla_m(pg, sum, val, b_vals_vec);
    }
    svst1(pg, &y[b * SIMD_WIDTH], sum);
  }
}

void mult_ell_s(struct ell *ell, double *x, double *y)
{
    int r = ell->rows - ell->num_blocks * SIMD_WIDTH; // remainder rows
    for (int b = 0; b < ell->num_blocks; b++) {
        double t[SIMD_WIDTH];
        for (int s = 0; s < SIMD_WIDTH; s++) t[s] = 0;
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH) {
            for (int s = 0; s < SIMD_WIDTH; s++) {
                t[s] += x[ell->j[l + s]] * ell->A[l + s];
            }
        }
        if (b == ell->num_blocks - 1 && r > 0) {
            for (int s = 0; s < r; s++) y[b * SIMD_WIDTH + s] = t[s];
        } else {
            for (int s = 0; s < SIMD_WIDTH; s++) y[b * SIMD_WIDTH + s] = t[s];
        }
    }
}


struct ell *create_ell(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct ell *ell = SPMV_ALLOC(struct ell, 1);
    ell->rows = rows;
    ell->columns = columns;
    ell->nnz = nnz;
    int num_blocks = rows / SIMD_WIDTH;
    if (rows % SIMD_WIDTH) num_blocks++;
    ell->num_blocks = num_blocks;
    ell->block_ptr = SPMV_ALLOC(int, num_blocks + 1);
    ell->memusage = sizeof(int) * (num_blocks + 1);

    // sort as for CSR
    sort_mtx(nnz, mtx, by_row_mtx);

    // build temporary CSR row index
    int *row_ptr = SPMV_ALLOC(int, num_blocks * SIMD_WIDTH + 1);
    row_ptr[0] = 0;
    for (int l = 0, k = 0; k < num_blocks * SIMD_WIDTH; k++) {
        while (l < nnz && mtx[l].i == k) l++;
        row_ptr[k + 1] = l;
    }

    // compute space required with padding
    ell->block_ptr[0] = 0;
    for (int p = 0, b = 0; b < num_blocks; b++) {
        int max_length = 0;
        for (int s = 0; s < SIMD_WIDTH; s++) {
            int row_len = row_ptr[b * SIMD_WIDTH + s + 1] -
                          row_ptr[b * SIMD_WIDTH + s];
            if (row_len > max_length) max_length = row_len;
        }
        p += max_length;
        ell->block_ptr[b + 1] = p * SIMD_WIDTH;
    }
    int l = ell->block_ptr[num_blocks];
    ell->j = SPMV_ALLOC(int64_t, l);
    ell->A = SPMV_ALLOC(double, l);
    ell->memusage += (sizeof(int) + sizeof(double)) * l;

    // fill matrix
    for (int b = 0; b < num_blocks; b++) {
        int row_len[SIMD_WIDTH];
        for (int s = 0; s < SIMD_WIDTH; s++) {
            row_len[s] = row_ptr[b * SIMD_WIDTH + s + 1] -
                         row_ptr[b * SIMD_WIDTH + s];
        }
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH ) {
            for (int s = 0; s < SIMD_WIDTH; s++) {
                if (row_len[s] == 0) { // insert padding
                    ell->j[l + s] = 0;
                    ell->A[l + s] = 0.0;
                } else {
                    ell->j[l + s] = mtx[row_ptr[b * SIMD_WIDTH + s]].j;
                    ell->A[l + s] = mtx[row_ptr[b * SIMD_WIDTH + s]].a;
                    row_ptr[b * SIMD_WIDTH + s]++;
                    row_len[s]--;
                }
            }
        }
    }

    free(row_ptr);
    return ell;
}
