#include "spmv.h"

#if defined(__GNUC__) && defined(__x86_64__)
#define SIMD_WIDTH 4
typedef int v4si __attribute__ ((vector_size (16)));
typedef double v4df __attribute__ ((vector_size (32)));

void mult_ell_sort(struct ell_sort *ell, double *x, double *y)
{
    int r = ell->rows - ell->num_blocks * SIMD_WIDTH; // remainder rows
    #pragma omp parallel for
    for (int b = 0; b < ell->num_blocks; b++) {
        v4df s = { 0 };
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH) {
            v4si vj = *(v4si *)(ell->j + l);
            v4df vx; for (int k = 0; k < SIMD_WIDTH; k++) vx[k] = x[vj[k]];
            v4df va = *(v4df *)(ell->A + l);
            s += vx * va;
        }
        if (b == ell->num_blocks - 1 && r > 0) {
            for (int k = 0; k < r ; k++) y[ell->perm[b * SIMD_WIDTH + k]] = s[k];
        } else {
            for (int k = 0; k < SIMD_WIDTH; k++) y[ell->perm[b * SIMD_WIDTH + k]] = s[k];
        }
    }
}

#else
#define SIMD_WIDTH 4

void mult_ell_sort(struct ell_sort *ell, double *x, double *y)
{
    int r = ell->rows - ell->num_blocks * SIMD_WIDTH; // remainder rows
    #pragma omp parallel for
    for (int b = 0; b < ell->num_blocks; b++) {
        double t[SIMD_WIDTH];
        for (int s = 0; s < SIMD_WIDTH; s++) t[s] = 0;
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH) {
            #pragma omp simd
            for (int s = 0; s < SIMD_WIDTH; s++) {
                t[s] += x[ell->j[l + s]] * ell->A[l + s];
            }
        }
        if (b == ell->num_blocks - 1 && r > 0) {
            #pragma omp simd
            for (int s = 0; s < r; s++) y[ell->perm[b * SIMD_WIDTH + s]] = t[s];
        } else {
            #pragma omp simd
            for (int s = 0; s < SIMD_WIDTH; s++) y[ell->perm[b * SIMD_WIDTH + s]] = t[s];
        }
    }
}

#endif

struct ell_row {
    int id;
    int ptr;
    int length;
};

static int by_row_length(const void *pa, const void *pb)
{
    const struct ell_row *a = (struct ell_row *)pa;
    const struct ell_row *b = (struct ell_row *)pb;
    if (a->length < b->length) return 1;
    if (a->length > b->length) return -1;
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

struct ell_sort *create_ell_sort(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct ell_sort  *ell = SPMV_ALLOC(struct ell_sort, 1);
    ell->rows = rows;
    ell->columns = columns;
    ell->nnz = nnz;
    int num_blocks = rows / SIMD_WIDTH;
    if (rows % SIMD_WIDTH) num_blocks++;
    ell->num_blocks = num_blocks;
    ell->perm = SPMV_ALLOC(int, rows);
    ell->block_ptr = SPMV_ALLOC(int, num_blocks + 1);
    ell->memusage = sizeof(int) * (num_blocks + 1 + rows);

    // sort as for CSR
    sort_mtx(nnz, mtx, by_row_mtx);

    // build temporary CSR row index
    struct ell_row *row = SPMV_ALLOC(struct ell_row, rows);
    for (int l = 0, k = 0; k < rows; k++) {
        row[k].id = k;
        row[k].ptr = l;
        while (l < nnz && mtx[l].i == k) l++;
        row[k].length = l - row[k].ptr;
    }

    // compute row permutation
    qsort(row, rows, sizeof(struct ell_row), by_row_length);
    for (int k = 0; k < rows; k++) ell->perm[k] = row[k].id;

    // compute space required with padding
    ell->block_ptr[0] = 0;
    for (int p = 0, b = 0; b < num_blocks; b++) {
        int max_length = 0;
        for (int s = 0; s < SIMD_WIDTH; s++) {
            if (b * SIMD_WIDTH >= rows) break;
            int rl = row[b * SIMD_WIDTH + s].length;
            if (rl > max_length) max_length = rl;
        }
        p += max_length;
        ell->block_ptr[b + 1] = p * SIMD_WIDTH;
    }
    int l = ell->block_ptr[num_blocks];
    ell->j = SPMV_ALLOC(int, l);
    ell->A = SPMV_ALLOC(double, l);
    ell->memusage += (sizeof(int) + sizeof(double)) * l;

    // fill matrix
    for (int b = 0; b < num_blocks; b++) {
        int row_len[SIMD_WIDTH];
        for (int s = 0; s < SIMD_WIDTH; s++) {
            row_len[s] = b * SIMD_WIDTH + s < rows ? row[b * SIMD_WIDTH + s].length : 0;
        }
        for (int l = ell->block_ptr[b]; l < ell->block_ptr[b + 1]; l += SIMD_WIDTH ) {
            for (int s = 0; s < SIMD_WIDTH; s++) {
                if (row_len[s] == 0) { // insert padding
                    ell->j[l + s] = 0;
                    ell->A[l + s] = 0.0;
                } else {
                    ell->j[l + s] = mtx[row[b * SIMD_WIDTH + s].ptr].j;
                    ell->A[l + s] = mtx[row[b * SIMD_WIDTH + s].ptr].a;
                    row[b * SIMD_WIDTH + s].ptr++;
                    row_len[s]--;
                }
            }
        }
    }

    free(row);
    return ell;
}
