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


#define PREFETCH_1D_STRIDED(D, N, S)                                \
    __asm volatile("li  t0, 1        # set count_h=1        \n"     \
                   "sll t1, %[n], 5  # set count_v=N[4:0]   \n"     \
                   "or  t0, t0, t1                          \n"     \
                   "li  t1, 0x1C00   # set enable bits      \n"     \
                   "or  t0, t0, t1                          \n"     \
                   "sll t1, %[s], 16 # set stride=S[47:0]   \n"     \
                   "or  t0, t0, t1                          \n"     \
                   "mv  t1, %[d]     # move D to t1         \n"     \
                   ".word 0x00532033 # prefetch.sh t0, t1   \n"     \
                   :                                                \
                   : [d] "r" (D), [n] "r" (N), [s] "r" (S)          \
                   : "t0", "t1"                                     \
                   );

#ifdef EPI

#ifdef EPI_EXT_07

void sellcs_mv_d(const sellcs_matrix_t* matrix,
    const elem_t* restrict x,
    elem_t* restrict y,
    const index_t start_slice,
    const index_t end_slice)
{
    const uint32_t vlen = matrix->C;

    __epi_1xi64 idxmult = __builtin_epi_vbroadcast_1xi64(8, vlen);

    for (size_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        index_t row_idx = slice_idx << matrix->shift;
        unsigned long int max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
        unsigned long int gvl = 0;

        gvl = __builtin_epi_vsetvl(max_lanes, __epi_e64, __epi_m1);
				asm volatile("vor.vv v4, v4 ,v4\n");


        elem_t* values_pointer = &matrix->values[matrix->slice_pointers[slice_idx]];
        index_t* colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

        // This changes in v0.9
        __epi_1xf64 tmp_results = __builtin_epi_vbroadcast_1xf64(0.0, gvl);

        index_t swidth = matrix->slice_widths[slice_idx];
        // index_t act_lanes_idx = matrix->vop_pointers[slice_idx];

    #ifndef INDEX64
    #else
        __epi_1xi64 y_sc_idx = __builtin_epi_vload_1xi64(&matrix->row_order[row_idx], gvl);
    #endif
        y_sc_idx = __builtin_epi_vmul_1xi64(y_sc_idx, idxmult, gvl);

        for (index_t i = 0; i < swidth; i++) {
            // Load Values and Column indices
            __epi_1xf64 values_vblock = __builtin_epi_vload_1xf64(values_pointer, gvl);

    #ifndef INDEX64
    #else
            __epi_1xi64 col_index_vblock = __builtin_epi_vload_1xi64(colidx_pointer, gvl);
    #endif

//						PREFETCH_1D_STRIDED(values_pointer+vlen, gvl>>3, 1);
//						PREFETCH_1D_STRIDED(colidx_pointer+vlen, gvl>>3, 1);


            col_index_vblock = __builtin_epi_vmul_1xi64(col_index_vblock, idxmult, gvl);
            // Gather X
            __epi_1xf64 x_vblock = __builtin_epi_vload_indexed_1xf64(&x[0], col_index_vblock, gvl);
            //__epi_1xf64 x_vblock = __builtin_epi_vload_nt_indexed_1xf64(&x[0], col_index_vblock, gvl);
            //__epi_1xf64 x_vblock = __builtin_epi_vload_ext_indexed_1xf64(&x[0], col_index_vblock, __epi_nt ,gvl);


            // Multiply; this changes in 0.9
            /* Read carefully:
               If DFC optimization is enabled; The lanes mutiplying and accumulated are defined by act_lanes
               act_lanes specifies a shorter or equal vector length each iteration.
               
               EPI Vector extension specifies that lanes beyond the VLEN requested
               to execute the instruction are set to zero. Therefore, a shorter operation will
               delete the results accumulated in the register by other previous operations with longer VLEN.

               We must use the GVL when accumulating.
               
               In 0.9 the value of the elements beyond the VLEN requested is undefined.
            */
            tmp_results = __builtin_epi_vfmacc_1xf64(tmp_results, x_vblock, values_vblock, gvl);

            values_pointer += vlen;
            colidx_pointer += vlen;
        }
        __builtin_epi_vstore_indexed_1xf64(&y[0], tmp_results, y_sc_idx, gvl);
    }
}

#else


void sellcs_mv_d(const sellcs_matrix_t* matrix,
                 const elem_t* restrict x,
                 elem_t* restrict y,
                 const index_t start_slice,
                 const index_t end_slice)
{
    const uint32_t vlen = matrix->C;
    // fprintf(stderr, "This version has not been tested\n");

    // New instruction used in 0.9
    __epi_1xi64 idxmult = __builtin_epi_vmv_v_x_1xi64(8, vlen);

    for (size_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        index_t row_idx = slice_idx << matrix->shift;
        unsigned long int max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
        unsigned long int gvl = 0;

        gvl = __builtin_epi_vsetvl(max_lanes, __epi_e64, __epi_m1);

        elem_t* values_pointer = &matrix->values[matrix->slice_pointers[slice_idx]];
        index_t* colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

        // This changes in v0.9
        __epi_1xf64 tmp_results = __builtin_epi_vfmv_v_f_1xf64(0.0, gvl);

        index_t swidth = matrix->slice_widths[slice_idx];

    #ifndef INDEX64
        __epi_1xi32 y_sc_idx = __builtin_epi_vload_1xi32(&matrix->row_order[row_idx], gvl);
    #else
        // __epi_1xi64 y_sc_idx = __builtin_epi_vload_unsigned_1xi64(&matrix->row_order[row_idx], gvl);
        __epi_1xi64 y_sc_idx = __builtin_epi_vload_1xi64(&matrix->row_order[row_idx], gvl);
    #endif
        y_sc_idx = __builtin_epi_vmul_1xi64(y_sc_idx, idxmult, gvl);

        for (index_t i = 0; i < swidth; i++) {
            // Load Values and Column indices
            __epi_1xf64 values_vblock = __builtin_epi_vload_1xf64(values_pointer, gvl);

    #ifndef INDEX64
            __epi_1xi32 col_index_vblock = __builtin_epi_vload_1xi32(colidx_pointer, gvl);
    #else
            __epi_1xi64 col_index_vblock = __builtin_epi_vload_1xi64(colidx_pointer, gvl);
    #endif

            col_index_vblock = __builtin_epi_vmul_1xi64(col_index_vblock, idxmult, gvl);
            // Gather X
            __epi_1xf64 x_vblock = __builtin_epi_vload_indexed_1xf64(&x[0], col_index_vblock, gvl);

            tmp_results = __builtin_epi_vfmacc_1xf64(tmp_results, x_vblock, values_vblock, gvl);

            values_pointer += vlen;
            colidx_pointer += vlen;
        }
        __builtin_epi_vstore_indexed_1xf64(&y[0], tmp_results, y_sc_idx, gvl);
    }
}



// void kernel_sellcs_dfc_epi(const SparseMatrixSELLCS* restrict matrix,
//     const elem_t* restrict x,
//     elem_t* restrict y,
//     const size_t start_slice,
//     const size_t end_slice)
// {
//     const uint32_t vlen = matrix->C;

//     __epi_1xi64 idxmult =  __builtin_epi_vmv_v_x_1xi64(8, vlen);

//     for (size_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
//         size_t row_idx = slice_idx << 6;
//         unsigned long int max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
//         unsigned long int gvl = 0;

//         gvl = __builtin_epi_vsetvl(max_lanes, __epi_e64, __epi_m1);

//         elem_t* values_pointer = &matrix->values[matrix->slice_pointers[slice_idx]];
//         uint64_t* colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

//         __epi_1xf64 tmp_results = __builtin_epi_vfmv_v_f_1xf64(0.0, gvl);

//         size_t swidth = matrix->slice_widths[slice_idx];
//         size_t act_lanes_idx = matrix->vop_pointers[slice_idx];

//         __epi_1xi64 y_sc_idx = __builtin_epi_vload_unsigned_1xi64(&matrix->row_order[row_idx], gvl);
//         y_sc_idx = __builtin_epi_vmul_1xi64(y_sc_idx, idxmult, gvl);

//         for (size_t i = 0; i < swidth; i++) {
//             unsigned long int act_lanes = (unsigned long int) matrix->vop_lengths[act_lanes_idx++] + 1;
//             // Load Values and Column indices
//             __epi_1xf64 values_vblock = __builtin_epi_vload_1xf64(values_pointer, act_lanes);
//             __epi_1xi64 col_index_vblock = __builtin_epi_vload_unsigned_1xi64(colidx_pointer, act_lanes);
//             col_index_vblock = __builtin_epi_vmul_1xi64(col_index_vblock, idxmult, act_lanes);
//             // Gather X
//             __epi_1xf64 x_vblock = __builtin_epi_vload_indexed_1xf64(&x[0], col_index_vblock, act_lanes);

//             // Multiply; this was changed from in 0.7
//             /* Read carefully:
//                In 0.9 the value of the elements beyond the VLEN requested is undefined. We can use act_lanes.
               
//                Possible danger identified by Roger:
//                In the operation
//                               __epi_1xf64 result = __builtin_epi_vfmacc_1xf64(c, a, b, vlen)

//                The compiler will detect that the variable result is overwritten.
//                Even if 'result' and 'c' is the same variable, the register allocator
//                might decide to use different registers producing wrong results.
               

//             */
//             tmp_results = __builtin_epi_vfmacc_1xf64(tmp_results, x_vblock, values_vblock, act_lanes);

//             values_pointer += act_lanes;
//             colidx_pointer += act_lanes;
//         }

//         __builtin_epi_vstore_indexed_1xf64(&y[0], tmp_results, y_sc_idx, gvl);
//     }
// }

#endif

#endif
