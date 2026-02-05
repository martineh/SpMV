#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_sve.h>

#include "ginkgo_spmv.h"


#define SELLCS_BLOCK 4

void mult_sellcs(struct sellcs *sellcs, double *x, double *y) {
    const svbool_t pg = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    svfloat64_t va, vx, vprod;
    svint64_t vidx;
    size_t vl;
    for (int b = 0; b < sellcs->num_blocks; b++) {
        svfloat64_t sum = zeros;
        for (int l = sellcs->block_ptr[b]; l < sellcs->block_ptr[b + 1]; l += SELLCS_BLOCK) {
            svfloat64_t val        = svld1(pg, &sellcs->A[l]);
            svint64_t col          = svld1_s64(pg, &sellcs->j[l]);
            svfloat64_t b_vals_vec = svld1_gather_index(pg, x, col);
            sum                    = svmla_m(pg, sum, val, b_vals_vec);
        }
        svst1(pg, &y[b * SELLCS_BLOCK], sum);
    }
}

int by_row_mtx(const struct mtx_coo *a, const struct mtx_coo *b);


struct sellcs *create_sellcs(int rows, int columns, int nnz, int *i, int *j, double *A) {
    //struct sellcs *sellcs = SPMV_ALLOC(struct sellcs, 1);
    struct sellcs *sellcs = (struct sellcs *)malloc(sizeof(struct sellcs));

    sellcs->rows = rows;
    sellcs->columns = columns;
    sellcs->nnz = nnz;
    int num_blocks = rows / SELLCS_BLOCK;
    if (rows % SELLCS_BLOCK) num_blocks++;
    sellcs->num_blocks = num_blocks;

    //sellcs->block_ptr = SPMV_ALLOC(int, num_blocks + 1);
    sellcs->block_ptr = (int *)malloc(sizeof(int) * (num_blocks + 1));
    sellcs->memusage  = sizeof(int) * (num_blocks + 1);

    // create mtx
    struct mtx_coo *mtx = (struct mtx_coo*)malloc(sizeof(struct mtx_coo) * nnz);
    int pos = 0;
    for (int r = 0; r < rows; r++) {
        for (int k = i[r]; k < i[r+1]; k++) {
            mtx[pos].i = r;
            mtx[pos].j = j[k];
            mtx[pos].a = A[k];
            pos++;
        }
    }

    //for (int ii = 0; ii < nnz; ii++) {
      //mtx[ii].i = i[ii]; mtx[ii].j = j[ii]; mtx[ii].a = A[ii]; 
    //}

    //--------------------------------------------------------
    // sort as for CSR
    //--------------------------------------------------------
    //sort_mtx(nnz, mtx, by_row_mtx);
    bool sorted = true;
    int ii = 1;
    while (sorted && ii < nnz) {
        int c = by_row_mtx(&mtx[ii - 1], &mtx[ii]);
        if (c == 1) {
            sorted = false;
        }
        ii++;
    }
    if (!sorted) {
        qsort(mtx, nnz, sizeof(struct mtx_coo), (int (*)(const void *, const void *))by_row_mtx);
    }
    //--------------------------------------------------------

    // build temporary CSR row index
    //int *row_ptr = SPMV_ALLOC(int, num_blocks * SELLCS_BLOCK + 1);
    int *row_ptr = (int *)malloc(sizeof(int) * (num_blocks * SELLCS_BLOCK + 1));

    row_ptr[0] = 0;
    for (int l = 0, k = 0; k < num_blocks * SELLCS_BLOCK; k++) {
        while (l < nnz && mtx[l].i == k) l++;
        row_ptr[k + 1] = l;
    }

    // compute space required with padding
    sellcs->block_ptr[0] = 0;
    for (int p = 0, b = 0; b < num_blocks; b++) {
        int max_length = 0;
        for (int s = 0; s < SELLCS_BLOCK; s++) {
            int row_len = row_ptr[b * SELLCS_BLOCK + s + 1] -
                          row_ptr[b * SELLCS_BLOCK + s];
            if (row_len > max_length) max_length = row_len;
        }
        p += max_length;
        sellcs->block_ptr[b + 1] = p * SELLCS_BLOCK;
    }
    int l = sellcs->block_ptr[num_blocks];
    //sellcs->j = SPMV_ALLOC(int64_t, l);
    //sellcs->A = SPMV_ALLOC(double, l);
    sellcs->j = (int64_t *)malloc(sizeof(int64_t) * l);
    sellcs->A = (double  *)malloc(sizeof(double)  * l);
    sellcs->memusage += (sizeof(int) + sizeof(double)) * l;

    // fill matrix
    for (int b = 0; b < num_blocks; b++) {
        int row_len[SELLCS_BLOCK];
        for (int s = 0; s < SELLCS_BLOCK; s++) {
            row_len[s] = row_ptr[b * SELLCS_BLOCK + s + 1] -
                         row_ptr[b * SELLCS_BLOCK + s];
        }
        for (int l = sellcs->block_ptr[b]; l < sellcs->block_ptr[b + 1]; l += SELLCS_BLOCK ) {
            for (int s = 0; s < SELLCS_BLOCK; s++) {
                if (row_len[s] == 0) { // insert padding
                    sellcs->j[l + s] = 0;
                    sellcs->A[l + s] = 0.0;
                } else {
                    sellcs->j[l + s] = mtx[row_ptr[b * SELLCS_BLOCK + s]].j;
                    sellcs->A[l + s] = mtx[row_ptr[b * SELLCS_BLOCK + s]].a;
                    row_ptr[b * SELLCS_BLOCK + s]++;
                    row_len[s]--;
                }
            }
        }
    }

    free(row_ptr);
    return sellcs;
}

