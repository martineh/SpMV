#include "spmv.h"

#define MAX_WIDTH 256

void mult_ell0(struct ell0 *ell0, double *x, double *y)
{
    double buffer[ell0->threads - 1];
    #pragma omp parallel num_threads(ell0->threads)
    {
        int t = omp_get_thread_num();
        struct ell0_thread *ell = ell0->p + t;
        double s[ell->rows];
        #pragma omp simd
        for (int w = 0; w < ell->rows; w++) s[w] = 0;
#if 0
        for (int b = 0; b < ell->nb; b++) {
            int *j = ell->j + ell->block_ptr[b];
            double *A = ell->A + ell->block_ptr[b];
            switch (ell->block_width[b]) {
            case 1:
                double t = 0;
                #pragma omp simd
                for (int l = 0; l < ell->block_len[b]; l++) {
                    t += x[*j++] * *A++;
                }
                s[0] += t;
                break;
            default:
                for (int l = 0; l < ell->block_len[b]; l++) {
                    #pragma omp simd
                    for (int w = 0; w < ell->block_width[b]; w++) {
                        s[w] += x[*j++] * *A++;
                    }
                }
            }
        }
#else
        int *j = ell->j;
        double *A = ell->A;
        for (int k = 0; k < ell->block_width[ell->nb - 1]; k += MAX_WIDTH) {
            for (int b = 0; b < ell->nb; b++) {
                for (int l = 0; l < ell->block_len[b]; l++) {
                    int c = ell->block_width[b] - k;
                    if (c > 0) {
                        if (c > MAX_WIDTH ) c = MAX_WIDTH;
                        #pragma omp simd
                        for (int w = 0; w < c; w++) {
                            s[k + w] += x[*j++] * *A++;
                        }
                    }
                }
            }
        }
#endif
        int *r = ell->row_id;
        if (ell->shared_row != -1) {
            #pragma omp simd
            for (int w = 0; w < ell->shared_row; w++) {
                y[r[w]] = s[w];
            }
            // this row is shared with the previous thread
            buffer[t - 1] = s[ell->shared_row];
        }
        #pragma omp simd
        for (int w = ell->shared_row + 1; w < ell->rows; w++) {
            y[r[w]] = s[w];
        }
    }
    // update partial rows
    for (int t = 1; t < ell0->threads; t++) {
        int k = ell0->p[t].shared_row;
        if (k != -1) {
            y[ell0->p[t].row_id[k]] += buffer[t - 1];
        }
    }
}

struct aux_row {
    int id;
    int ptr;
    int len;
};

static int sort_aux_row(const void *pa, const void *pb)
{
    const struct aux_row *a = (struct aux_row *)pa;
    const struct aux_row *b = (struct aux_row *)pb;
    if (a->len < b->len) return 1;
    if (a->len > b->len) return -1;
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

static void create_ell0_thread(struct ell0_thread *ell, int nnz, struct mtx *mtx)
{
    // count rows
    int rows = 1;
    for (int l = 1; l < nnz; l++) {
        if (mtx[l].i != mtx[l - 1].i) rows++;
    }

    // sort rows by their lenght
    struct aux_row *aux = SPMV_ALLOC(struct aux_row, rows);
    for (int l = 0, k = 0; k < rows; k++) {
        aux[k].id = mtx[l].i;
        aux[k].ptr = l;
        while (l < nnz && mtx[l].i == aux[k].id) l++;
        aux[k].len = l - aux[k].ptr;
    }
    qsort(aux, rows, sizeof(struct aux_row), sort_aux_row);

    // compute number of blocks
    int nb = 0;
    for (int k = 0; k < rows; ) {
        int l = aux[k].len;
        while (k < rows && l == aux[k].len) k++;
        nb++;
    }

    // allocate memory
    ell->rows = rows;
    ell->nb = nb;
    ell->block_ptr = SPMV_ALLOC(int, nb);
    ell->block_len = SPMV_ALLOC(int, nb);
    ell->block_width = SPMV_ALLOC(int, nb);
    ell->row_id = SPMV_ALLOC(int, rows);
    ell->j = SPMV_ALLOC(int, nnz);
    ell->A = SPMV_ALLOC(double, nnz);

    for (int k = 0; k < rows; k++) ell->row_id[k] = aux[k].id;

    // compute block sizes
    for (int b = 0, k = 0; k < rows; ) {
        int l = aux[k].len;
        while (k < rows && l == aux[k].len) k++;
        ell->block_len[b] = l;
        ell->block_width[b] = k;
        b++;
    }
    for (int b = 0; b < nb - 1; b++) {
        ell->block_len[b] -= ell->block_len[b + 1];
    }

    // fill matrix
#if 0
    for (int k = 0, b = 0; b < nb; b++) {
        ell->block_ptr[b] = k;
        for (int l = 0; l < ell->block_len[b]; l++) {
            for (int w = 0; w < ell->block_width[b]; w++) {
                int p = aux[w].ptr;
                ell->j[k] = mtx[p].j;
                ell->A[k] = mtx[p].a;
                aux[w].ptr++;
                k++;
            }
        }
    }
#else
    int *j = ell->j;
    double *A = ell->A;
    for (int k = 0; k < ell->block_width[ell->nb - 1]; k += MAX_WIDTH) {
        for (int b = 0; b < ell->nb; b++) {
            for (int l = 0; l < ell->block_len[b]; l++) {
                int c = ell->block_width[b] - k;
                if (c > 0) {
                    if (c > MAX_WIDTH ) c = MAX_WIDTH;
                    #pragma omp simd
                    for (int w = 0; w < c; w++) {
                        int p = aux[k + w].ptr;
                        *j++ = mtx[p].j;
                        *A++ = mtx[p].a;
                        aux[k + w].ptr++;
                    }
                }
            }
        }
    }
#endif

    free(aux);
}

struct ell0 *create_ell0(int rows, int columns, int nnz, struct mtx *mtx)
{
    int threads = omp_get_max_threads();
    int struct_size = sizeof(struct ell0) + sizeof(struct ell0_thread) * threads;
    struct ell0 *ell = (struct ell0 *)malloc(struct_size);
    ell->memusage = struct_size;
    ell->rows = rows;
    ell->columns = columns;
    ell->nnz = nnz;
    ell->threads = threads;

    // sort as for CSR
    sort_mtx(nnz, mtx, by_row_mtx);

    // parallel initialization
    int first_row[threads];
    int last_row[threads];
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num();
        int length = (nnz + t) / threads; // number of non-zeros
        int start = 0; // first non-zero
        for (int k = 0; k < t; k++) start += (nnz + k) / threads;
        first_row[t] = mtx[start].i;
        last_row[t] = mtx[start + length - 1].i;
        create_ell0_thread(ell->p + t, length, mtx + start);
    }

    // flag shared rows among threads
    ell->p[0].shared_row = -1;
    for (int t = 1; t < threads; t++) {
        if (first_row[t] == last_row[t - 1]) {
            int k = 0;
            while (k < ell->p[t].rows && ell->p[t].row_id[k] != first_row[t]) k++;
            ell->p[t].shared_row = k;
        } else {
            ell->p[t].shared_row = -1;
        }
    }

    for (int t = 0; t < threads; t++) {
        ell->memusage += sizeof(int) * (3 * ell->p[t].nb + ell->p[t].rows);
    }
    ell->memusage += (sizeof(int) + sizeof(double)) * nnz;
    return ell;
}
