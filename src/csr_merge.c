#include "spmv.h"

void searchPathOnDiag(int diagonal, int* list_A, int nor, int nz, int* list_a_coord, int* list_b_coord);

#define min(a,b) (((a)<(b))?(a):(b))

    /**
     * @brief Merge-based parallel SpMVM algorithm based on CRS format
     *
     * @param A Matrix
     * @param x Input vector
     * @param y Output vector
     * row_carry and value_carry of dimension nthreads, allocated prior to call
     */
    void spmvmCRSMerge(int n, int nz, int* row_offsets, int* col_idxs, double* values, double* x, double* y, int nthreads, int* row_carry, double* value_carry) {

        int* row_end_offsets = row_offsets + 1; // Merge list A
        int num_merge_items = n + nz;
        int items_per_thread = (num_merge_items + nthreads - 1) / nthreads;

        // Parallel section: each thread processes 1 loop iteration
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int tid = 0; tid < nthreads; ++tid) {
            // Find start and stop coordinates of merge path
            int diagonal = min(items_per_thread*tid, num_merge_items);
            int diagonal_end = min(diagonal + items_per_thread, num_merge_items);

            int thread_a_coord, thread_b_coord, thread_a_coord_end, thread_b_coord_end;
            searchPathOnDiag(diagonal, row_end_offsets, n, nz, &thread_a_coord, &thread_b_coord);
            searchPathOnDiag(diagonal_end, row_end_offsets, n, nz, &thread_a_coord_end, &thread_b_coord_end);

            // Consume merge items for every whole row
            double running_total = 0.;
            for (; thread_a_coord < thread_a_coord_end; ++thread_a_coord) {
                for (; thread_b_coord < row_end_offsets[thread_a_coord]; ++thread_b_coord) {
                    running_total += values[thread_b_coord]*x[col_idxs[thread_b_coord]];
                }

                y[thread_a_coord] = running_total;
                running_total = 0.;
            }

            // Consume last row (if its a partial row)
            for (; thread_b_coord < thread_b_coord_end; ++thread_b_coord) {
                running_total += values[thread_b_coord] * x[col_idxs[thread_b_coord]];
            }

            // Save carry results
            row_carry[tid] = thread_a_coord_end;
            value_carry[tid] = running_total;
        }

        // Fix the carry results
        for (int tid = 0; tid < nthreads; ++tid) {
            if (row_carry[tid] < n) {
                y[row_carry[tid]] += value_carry[tid];
            }
        }
    }


    void searchPathOnDiag(int diagonal, int* list_A, int nor, int nz,
                          int* list_a_coord, int* list_b_coord) {
        // Search range for A list
        int a_coord_min = 0;
        if (diagonal > nz) {
            a_coord_min = diagonal - nz;
        }
        int a_coord_max = min(diagonal, nor);

        // Binary-search along the diagonal
        int pivot;
        while (a_coord_min < a_coord_max) {
            pivot = (a_coord_min + a_coord_max) >> 1; // Is equal to middle of interval (division is shift for speed)

            if (list_A[pivot] <= diagonal-pivot-1) {
                a_coord_min = pivot + 1; // Keep top-right of diag
            } else {
                a_coord_max = pivot; // Keep bottom-left of diag
            }
        }

        *list_a_coord = min(a_coord_min, nor);
        *list_b_coord = diagonal-a_coord_min;
    }

struct csr_merge *create_csr_merge(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct csr_merge *csr = SPMV_ALLOC(struct csr_merge, 1);
    csr->rows = rows;
    csr->columns = columns;
    csr->nnz = nnz;
    csr->nthreads = omp_get_max_threads();
    csr->row_carry = SPMV_ALLOC(int, csr->nthreads);
    csr->value_carry = SPMV_ALLOC(double, csr->nthreads);
    csr->row_ptr = SPMV_ALLOC(int, rows + 1);
    csr->col_ind = SPMV_ALLOC(int, nnz);
    csr->val = SPMV_ALLOC(double, nnz);
    csr->memusage = sizeof(int) * (rows + 1) + (sizeof(int) + sizeof(double)) * nnz +
                    (sizeof(int) + sizeof(double)) * csr->nthreads;

    sort_mtx(nnz, mtx, by_row_mtx);

    csr->row_ptr[0] = 0;
    for (int l = 0, k = 0; k < rows; k++) {
        while (l < nnz && mtx[l].i == k) {
            csr->col_ind[l] = mtx[l].j;
            csr->val[l] = mtx[l].a;
            l++;
        }
        csr->row_ptr[k + 1] = l;
    }

    return csr;
}

void mult_csr_merge(struct csr_merge *csr, double *x, double *y)
{
    spmvmCRSMerge(csr->rows, csr->nnz, csr->row_ptr, csr->col_ind, csr->val, x, y, csr->nthreads, csr->row_carry, csr->value_carry);
}
