#include "spmv.h"

#ifdef EPI

#define SIMD_WIDTH 256

void mult_jds(struct jds *jds, double *x, double *y)
{
    // first block
    int gvlmax = __builtin_epi_vsetvl(jds->rows, __epi_e64, __epi_m1);
    __epi_1xf64 t = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
    if (jds->num_row1 > 0) { // special case for first row
        __epi_1xf64 s = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
        int *J = jds->j + jds->jd_ptr[jds->num_diag[0]];
        double *A = jds->A + jds->jd_ptr[jds->num_diag[0]];
        int r = jds->num_row1 % gvlmax;
        if (r > 0) {
            int gvl = __builtin_epi_vsetvl(r, __epi_e64, __epi_m1);
            __epi_1xf64 a = __builtin_epi_vload_1xf64(A, gvl);
            __epi_1xi64 j = __builtin_epi_vload_1xi64(J, gvl);
            __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
            s = __builtin_epi_vfmacc_1xf64(s, a, v, gvl);
        }
        if (jds->num_row1 >= gvlmax) { // very long row
            for (; r < jds->num_row1; r += gvlmax) {
                __epi_1xf64 a = __builtin_epi_vload_1xf64(A, gvlmax);
                __epi_1xi64 j = __builtin_epi_vload_1xi64(J, gvlmax);
                __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvlmax);
                s = __builtin_epi_vfmacc_1xf64(s, a, v, gvlmax);
            }
            __epi_1xf64 z = __builtin_epi_vsetfirst_1xf64(0, gvlmax);
            t = __builtin_epi_vfredsum_1xf64(s, z, gvlmax);
        } else { // short row
            __epi_1xf64 z = __builtin_epi_vsetfirst_1xf64(0, gvl);
            t = __builtin_epi_vfredsum_1xf64(s, z, gvl);
        }
    }
    for (int d = jds->num_diag[b] - 1; d >= 0; d--) {
        int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d];
        int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
        __epi_1xf64 a = __builtin_epi_vload_1xf64(jds->A + jds->jd_ptr[d], gvl);
        __epi_1xi64 j = __builtin_epi_vload_1xi64(jds->j + jds->jd_ptr[d], gvl);
        __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
        t = __builtin_epi_vfmacc_1xf64(t, a, v, gvl);
    
    __epi_1xi64 j = __builtin_epi_vload_1xi64(jds->perm, gvlmax);
    __builtin_epi_vstore_indexed_1xf64(y, t, j, gvlmax);
    // rest of blocks
    for (int b = 1, k = SIMD_WIDTH; b < jds->num_blocks; b++) {
        int gvlmax = __builtin_epi_vsetvl(jds->rows - k, __epi_e64, __epi_m1);
        __epi_1xf64 t = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
        for (int d = jds->num_diag[b] - 1; d >= 0; d--) {
            int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d] - k;
            int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
            __epi_1xf64 a = __builtin_epi_vload_1xf64(jds->A + jds->jd_ptr[d] + k, gvl);
            __epi_1xi64 j = __builtin_epi_vload_1xi64(jds->j + jds->jd_ptr[d] + k, gvl);
            __epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
            t = __builtin_epi_vfmacc_1xf64(t, a, v, gvl);
        }
        __epi_1xi64 j = __builtin_epi_vload_1xi64(jds->perm + k, gvlmax);
        __builtin_epi_vstore_indexed_1xf64(y, t, j, gvlmax);
        k += gvlmax;
    }
}

#elif __RISCV__

#define SIMD_WIDTH 256

#include <riscv_vector.h>

void mult_jds(struct jds *jds, double *x, double *y) {
  //__epi_1xf64 t, a, v;
  //__epi_1xi64 j;
  vfloat64m4_t t, a, v;
  vuint32m2_t j;

  for (int k = 0; k < jds->rows;) {
    //int gvlmax = __builtin_epi_vsetvl(jds->rows - k, __epi_e64, __epi_m1);
    //t = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
    size_t  gvlmax = __riscv_vsetvl_e64m4(jds->rows - k);
    t = __riscv_vfmv_v_f_f64m4(0.0, gvlmax);
    for (int d = jds->num_diags - 1; d >= 0; d--) {
      int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d] - k;
      if (n <= 0) continue; // no more elements
      //int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
      //a = __builtin_epi_vload_1xf64(jds->A + jds->jd_ptr[d] + k, gvl);
      //j = __builtin_epi_vload_1xi64(jds->j + jds->jd_ptr[d] + k, gvl);
      //v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
      //t = __builtin_epi_vfmacc_1xf64(t, a, v, gvl);
      size_t gvl = __riscv_vsetvl_e64m4(n);
      a = __riscv_vle64_v_f64m4(jds->A + jds->jd_ptr[d] + k, gvl);
      j = __riscv_vle32_v_u32m2((unsigned int *)jds->j + jds->jd_ptr[d] + k, gvl);
      j = __riscv_vsll_vx_u32m2(j, 3, gvl);
      v = __riscv_vloxei32_v_f64m4(x, j, gvl);
      t = __riscv_vfmacc_vv_f64m4(t, a, v, gvl);
    }
    //j = __builtin_epi_vload_1xi64(jds->perm + k, gvlmax);
    //__builtin_epi_vstore_indexed_1xf64(y, t, j, gvlmax);
    j = __riscv_vle32_v_u32m2((unsigned int *)jds->perm + k, gvlmax);
    j = __riscv_vsll_vx_u32m2(j, 3, gvlmax);
    __riscv_vsoxei32_v_f64m4(y, j, t, gvlmax);
    k += gvlmax;
  }
}


/*
void mult_jds(struct jds *jds, double *x, double *y)
{
    // first block
    //int gvlmax = __builtin_epi_vsetvl(jds->rows, __epi_e64, __epi_m1);
    //__epi_1xf64 t = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
    size_t  gvlmax = __riscv_vsetvl_e64m4(jds->rows);
    vfloat64m4_t t = __riscv_vfmv_v_f_f64m4(0.0, gvlmax);

    if (jds->num_row1 > 0) { // special case for first row
        //__epi_1xf64 s = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
        vfloat64m4_t  s = __riscv_vfmv_v_f_f64m4(0.0, gvlmax);
        
	unsigned int *J = (unsigned int *)jds->j + jds->jd_ptr[jds->num_diag[0]];
        double       *A = jds->A + jds->jd_ptr[jds->num_diag[0]];
        int           r = jds->num_row1 % gvlmax;

        if (r > 0) {
            //int gvl = __builtin_epi_vsetvl(r, __epi_e64, __epi_m1);
            //__epi_1xf64 a = __builtin_epi_vload_1xf64(A, gvl);
            //__epi_1xi64 j = __builtin_epi_vload_1xi64(J, gvl);
            //__epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
            //s = __builtin_epi_vfmacc_1xf64(s, a, v, gvl);
            size_t  gvl = __riscv_vsetvl_e64m4(r);
	    vfloat64m4_t a = __riscv_vle64_v_f64m4(A, gvl);
	    vuint32m2_t   j = __riscv_vle32_v_u32m2(J, gvl);
	    j = __riscv_vsll_vx_u32m2(j, 3, gvl);
	    vfloat64m4_t v = __riscv_vloxei32_v_f64m4(x, j, gvl);
	    s = __riscv_vfmacc_vv_f64m4(s, a, v, gvl);
        }
        if (jds->num_row1 >= gvlmax) { // very long row
            for (; r < jds->num_row1; r += gvlmax) {
              //__epi_1xf64 a = __builtin_epi_vload_1xf64(A, gvlmax);
              //__epi_1xi64 j = __builtin_epi_vload_1xi64(J, gvlmax);
              //__epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvlmax);
              //s = __builtin_epi_vfmacc_1xf64(s, a, v, gvlmax);
	      vfloat64m4_t a = __riscv_vle64_v_f64m4(A, gvlmax);
	      vuint32m2_t   j = __riscv_vle32_v_u32m2(J, gvlmax);
	      j = __riscv_vsll_vx_u32m2(j, 3, gvlmax);
	      vfloat64m4_t   v = __riscv_vloxei32_v_f64m4(x, j, gvlmax);
	      s = __riscv_vfmacc_vv_f64m4(s, a, v, gvlmax);
            }
            //__epi_1xf64 z = __builtin_epi_vsetfirst_1xf64(0, gvlmax);
            //t = __builtin_epi_vfredsum_1xf64(s, z, gvlmax);
            vfloat64m1_t z = __riscv_vfmv_v_f_f64m1(0.0, gvlmax);
	    z = __riscv_vfredosum_vs_f64m4_f64m1(s, z, gvlmax);
	    double first = __riscv_vfmv_f_s_f64m1_f64(z);
	    t = __riscv_vfmv_v_f_f64m4(first, 1);
        } else { // short row
            //__epi_1xf64 z = __builtin_epi_vsetfirst_1xf64(0, gvl);
            //t = __builtin_epi_vfredsum_1xf64(s, z, gvl);
            vfloat64m1_t z = __riscv_vfmv_v_f_f64m1(0.0, jds->num_row1);
	    z = __riscv_vfredosum_vs_f64m4_f64m1(s, z, jds->num_row1);
	    double first = __riscv_vfmv_f_s_f64m1_f64(z);
	    t = __riscv_vfmv_v_f_f64m4(first, 1);
        }
    }
    
    for (int d = jds->num_diag[b] - 1; d >= 0; d--) {
        int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d];
        //int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
        //__epi_1xf64 a = __builtin_epi_vload_1xf64(jds->A + jds->jd_ptr[d], gvl);
        //__epi_1xi64 j = __builtin_epi_vload_1xi64(jds->j + jds->jd_ptr[d], gvl);
        //__epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
        //t = __builtin_epi_vfmacc_1xf64(t, a, v, gvl);
        size_t  gvl    = __riscv_vsetvl_e64m4(n);
	vfloat64m4_t a = __riscv_vle64_v_f64m4(jds->A + jds->jd_ptr[d], gvl);
	vuint32m4_t  j = __riscv_vle32_v_u32m2(jds->j + jds->jd_ptr[d], gvl);
	j = __riscv_vsll_vx_u32m2(j, 3, vl);
	vfloat64m4_t v = __riscv_vloxei32_v_f64m4(x, j, gvl);
	t = __riscv_vfmacc_vv_f64m4(t, a, v, gvl);
    }
    //__epi_1xi64 j = __builtin_epi_vload_1xi64(jds->perm, gvlmax);
    //__builtin_epi_vstore_indexed_1xf64(y, t, j, gvlmax);
    vuint32m2_t j = __riscv_vle32_v_u32m2(jds->perm, gvlmax);
    j = __riscv_vsll_vx_u32m2(j, 3, gvlmax);
    __riscv_vsoxei32_v_f64m4(y, j, t, gvlmax);

    // rest of blocks
    for (int b = 1, k = SIMD_WIDTH; b < jds->num_blocks; b++) {
      //int gvlmax = __builtin_epi_vsetvl(jds->rows - k, __epi_e64, __epi_m1);
      //__epi_1xf64 t = __builtin_epi_vbroadcast_1xf64(0.0, gvlmax);
      size_t  gvlmax = __riscv_vsetvl_e64m4(jds->rows - k);
      vfloat64m4_t t = __riscv_vfmv_v_f_f64m4(0.0, gvlmax);
      for (int d = jds->num_diag[b] - 1; d >= 0; d--) {
        int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d] - k;
        //int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
        //__epi_1xf64 a = __builtin_epi_vload_1xf64(jds->A + jds->jd_ptr[d] + k, gvl);
        //__epi_1xi64 j = __builtin_epi_vload_1xi64(jds->j + jds->jd_ptr[d] + k, gvl);
        //__epi_1xf64 v = __builtin_epi_vload_indexed_1xf64(x, j, gvl);
        //t = __builtin_epi_vfmacc_1xf64(t, a, v, gvl);
        size_t  gvl    = __riscv_vsetvl_e64m4(n);
	vfloat64m4_t a = __riscv_vle64_v_f64m4(jds->A + jds->jd_ptr[d] + k, gvl);
	vuint32m2_t  j = __riscv_vle32_v_u32m2(jds->j + jds->jd_ptr[d] + k, gvl);
	j = __riscv_vsll_vx_u32m2(j, 3, vl);
	vfloat64m4_t   v = __riscv_vloxei32_v_f64m4(x, j, gvl);
	t = __riscv_vfmacc_vv_f64m4(t, a, v, gvl);
      }
      //__epi_1xi64 j = __builtin_epi_vload_1xi64(jds->perm + k, gvlmax);
      //__builtin_epi_vstore_indexed_1xf64(y, t, j, gvlmax);
      vint32m2_t j = __riscv_vle64_v_i64m4(jds->perm + k, gvlmax);
      j = __riscv_vsll_vx_u32m2(j, 3, gvlmax);
      t = __riscv_vsoxei32_v_f64m4(y, j, t, gvlmax);

      k += gvlmax;
     *
    }
}
*/
#else

#define SIMD_WIDTH 256

void mult_jds(struct jds *jds, double *x, double *y)
{
    double t[SIMD_WIDTH];
    for (int b = 0, k = 0; b < jds->num_blocks; b++, k += SIMD_WIDTH) {
        #pragma omp simd
        for (int s = 0; s < SIMD_WIDTH; s++) t[s] = 0.0;
        if (k == 0 && jds->num_row1 > 0) { // first row
            int *j = jds->j + jds->jd_ptr[jds->num_diag[0]];
            double *A = jds->A + jds->jd_ptr[jds->num_diag[0]];
            #pragma omp simd
            for (int i = 0; i < jds->num_row1; i++) t[0] += x[j[i]] * A[i];
        }
        for (int d = 0; d < jds->num_diag[b]; d++) {
            int n = jds->jd_ptr[d + 1] - jds->jd_ptr[d] - k;
            int *j = jds->j + jds->jd_ptr[d] + k;
            double *A = jds->A + jds->jd_ptr[d] + k;
            if (n > SIMD_WIDTH) n = SIMD_WIDTH;
            #pragma omp simd
            for (int s = 0; s < n; s++) t[s] += x[j[s]] * A[s];
        }
        int r = jds->rows - k;
        if (r > SIMD_WIDTH) r = SIMD_WIDTH;
        #pragma omp simd
        for (int s = 0; s < r; s++) y[jds->perm[k + s]] = t[s];
    }
}

#endif

struct jds_row {
    int length;
    int id;
    int ptr;
};

static int by_row_length(const void *pa, const void *pb)
{
    const struct jds_row *a = (struct jds_row *)pa;
    const struct jds_row *b = (struct jds_row *)pb;
    if (a->length < b->length) return 1;
    if (a->length > b->length) return -1;
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

struct jds *create_jds(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct jds *jds = SPMV_ALLOC(struct jds, 1);
    jds->rows = rows;
    jds->columns = columns;
    jds->nnz = nnz;
#ifdef EPI
    jds->perm = SPMV_ALLOC(long, rows);
    jds->j = SPMV_ALLOC(long, nnz);
    jds->memusage = (sizeof(long) + sizeof(double)) * nnz +
                    sizeof(long) * rows;
#else
    jds->perm = SPMV_ALLOC(int, rows);
    jds->j = SPMV_ALLOC(int, nnz);
    jds->memusage = (sizeof(int) + sizeof(double)) * nnz +
                    sizeof(int) * rows;
#endif
    jds->A = SPMV_ALLOC(double, nnz);

    // sort as for CSR
    sort_mtx(nnz, mtx, by_row_mtx);

    // build temporary CSR row index and sort by row length
    struct jds_row *row = SPMV_ALLOC(struct jds_row, rows);
    for (int l = 0, k = 0; k < rows; k++) {
        row[k].id = k;
        row[k].ptr = l;
        while (l < nnz && mtx[l].i == k) l++;
        row[k].length = l - row[k].ptr;
    }
    qsort(row, rows, sizeof(struct jds_row), by_row_length);
    jds->num_diags = row[0].length;
    for (int k = 0; k < rows; k++) jds->perm[k] = row[k].id;
#if defined(EPI)
    for (int k = 0; k < rows; k++) jds->perm[k] *= 8;
#endif

    /*
     * Optimization over original JDS:
     * 1. Compute number of diagonals per SIMD block
     */
    jds->num_blocks = rows / SIMD_WIDTH;
    if (rows % SIMD_WIDTH) jds->num_blocks++;
    jds->num_diag = SPMV_ALLOC(int, jds->num_blocks);
    jds->memusage += sizeof(int) * jds->num_blocks;
    for (int k = 0; k < jds->num_blocks; k++) jds->num_diag[k] = row[k * SIMD_WIDTH].length;

    // fill matrix
    jds->jd_ptr = SPMV_ALLOC(int, jds->num_diags + 1);
    jds->memusage += sizeof(int) * (jds->num_diags + 1);
    jds->jd_ptr[0] = 0;
    for (int k = 0, l = 0; k < jds->num_diags; k++) {
        for (int r = 0; r < rows && row[r].length > 0; r++) {
#ifdef EPI
            jds->j[l] = mtx[row[r].ptr].j * 8;
#else
            jds->j[l] = mtx[row[r].ptr].j;
#endif
            jds->A[l] = mtx[row[r].ptr].a;
            row[r].ptr++;
            row[r].length--;
            l++;
        }
        jds->jd_ptr[k + 1] = l;
    }

    /*
     * Optimization over original JDS:
     * 2. Check if the first row is very long
     */
    jds->num_row1 = 0;
    for (int k = jds->num_diags - 1; k < rows; k++) {
        if (jds->jd_ptr[k + 1] - jds->jd_ptr[k] != 1) break;
        jds->num_row1++;
    }
    jds->num_diag[0] -= jds->num_row1;

    // compress diagonal pointers
    /* int col_len[jds->num_diags];
    for (int k = 0; k < jds->num_diags; k++) {
        col_len[k] = jds->jd_ptr[k + 1] - jds->jd_ptr[k];
    }

    int num_blocks = 1;
    for (int k = 1; k < jds->num_diags; k++) {
        if (col_len[k] != col_len[k - 1]) num_blocks++;
    }
    int block_height[num_blocks];
    int block_width[num_blocks];
    block_height[0] = col_len[0];
    block_width[0] = 1;

    for (int k = 1, l = 0; k < jds->num_diags; k++) {
        if (col_len[k] == col_len[k - 1]) {
            block_width[l]++;
        } else {
            l++;
            block_height[l] = col_len[k];
            block_width[l] = 1;
        }
    }
    printf("num_diags %d num_blocks %d\n", jds->num_diags, num_blocks);
    for (int k = 0; k < num_blocks; k++) {
        printf("block width %d height %d\n", block_width[k], block_height[k]);
    } */

    free(row);
    return jds;
}
