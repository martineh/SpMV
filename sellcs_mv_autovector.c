/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "sellcs-spmv.h"
#include "stdio.h"

void sellcs_mv_d_autovector(const sellcs_matrix_t* matrix,
    const elem_t* __restrict__ x,
    elem_t* __restrict__ y,
    const index_t start_row,
    const index_t end_row)
{
    const uint32_t vlen = matrix->C;
    for (size_t row_idx_i = start_row; row_idx_i < end_row; row_idx_i += vlen) {
        index_t slice_idx = row_idx_i >> matrix->shift;
        index_t row_idx = slice_idx << matrix->shift;

        // unsigned long int max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;

        elem_t* values_pointer = &matrix->values[matrix->slice_pointers[slice_idx]];
        index_t* colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

        index_t swidth = matrix->slice_widths[slice_idx];
        index_t* y_sc_idx = &matrix->row_order[row_idx];

        // Y is initialized to 0
        // Non-intrinsics version
        for (index_t j = 0; j < swidth; j++) {
            for (index_t k = 0; k < vlen; k++) {
                // Load Values and Column indices
                const elem_t aval = values_pointer[k];
                const index_t col_idx = colidx_pointer[k];
                const elem_t xval = x[col_idx];
                // if (j == 0)
                //     fprintf(stderr, "%lu,", y_sc_idx[k]);
                y[y_sc_idx[k]] += xval * aval;
            }
            values_pointer += vlen;
            colidx_pointer += vlen;
        }
        // fprintf(stderr, "\n");
    }
}
