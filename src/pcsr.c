#include <stdio.h>
#include "spmv.h"

void mult_pcsr(struct pcsr *pcsr, double *x, double *y)
{
    for (int i = 0; i < pcsr->rows; i++) y[i] = 0.0;
    for (int r = 0; r < pcsr->num_rows; r++) {
        double s = 0.0;
        #pragma omp simd
        for (int k = pcsr->row_start[r]; k < pcsr->row_start[r + 1]; k++) {
            s += x[pcsr->j1[r] + pcsr->j2[k]] * pcsr->A[k];
        }
        y[pcsr->i[r]] += s;
    }
}

void print_pcsr(struct pcsr *pcsr)
{
    for (int r = 0; r < pcsr->num_rows; r++) {
        for (int k = pcsr->row_start[r]; k < pcsr->row_start[r + 1]; k++) {
            printf("%d %d %d %e\n", pcsr->i[r], pcsr->j1[r],
                    pcsr->j2[k], pcsr->A[k]);
        }
    }
}

static int bs;

int by_split_row_mtx(const struct mtx *a, const struct mtx *b)
{
    if (a->j / bs < b->j / bs) return -1;
    if (a->j / bs > b->j / bs) return 1;
    if (a->i < b->i) return -1;
    if (a->i > b->i) return 1;
    if (a->j < b->j) return -1;
    if (a->j > b->j) return 1;
    return 0;
}

struct pcsr *create_pcsr(int rows, int columns, int nnz, struct mtx *mtx, int block_size)
{
    bs = block_size;
    sort_mtx(nnz, mtx, by_split_row_mtx);

    // count number of partitioned rows
    int n = 0;
    for (int k = 0; k < nnz; k++) {
        if (k == nnz - 1 || mtx[k].i != mtx[k + 1].i) {
            n++;
        }
    }

    struct pcsr *pcsr = SPMV_ALLOC(struct pcsr, 1);
    pcsr->rows = rows;
    pcsr->columns = columns;
    pcsr->nnz = nnz;
    pcsr->num_rows = n;
    pcsr->i = SPMV_ALLOC(int, n);
    pcsr->row_start = SPMV_ALLOC(int, n + 1);
    pcsr->j1 = SPMV_ALLOC(int, n);
    pcsr->j2 = SPMV_ALLOC(unsigned short, nnz);
    pcsr->A = SPMV_ALLOC(double, nnz);
    pcsr->memusage = 3 * sizeof(int) * n + (sizeof(short) + sizeof(double)) * nnz;

    pcsr->row_start[0] = 0;
    int k = 0;
    for (int l = 0; l < nnz; l++) {
        pcsr->j2[l] = mtx[l].j % bs;
        pcsr->A[l] = mtx[l].a;
        if (l == nnz - 1 || mtx[l].i != mtx[l + 1].i ||
                mtx[l].j / bs != mtx[l + 1].j / bs) {
            pcsr->i[k] = mtx[l].i;
            pcsr->j1[k] = (mtx[l].j / bs) * bs;
            pcsr->row_start[k + 1] = l + 1;
            k++;
        }
    }
    // print_pcsr(pcsr);

    return pcsr;
}
