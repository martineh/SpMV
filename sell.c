#include "spmv.h"
#include "sellcs-spmv.h"

uint64_t sigma_window = 16384;
int32_t max_vlen = 64;

#ifdef INDEX64
void *create_sell(struct csr_epi * csr)
#else
void *create_sell(struct csr * csr)
#endif
{
    // Init data structures
    sellcs_matrix_t *sellcs_matrix = SPMV_ALLOC(sellcs_matrix_t, 1);
    sellcs_init_params(max_vlen, sigma_window, sellcs_matrix);

#if defined(EPI) && defined(INDEX64)
    for (int k = 0; k < csr->nnz; k++) csr->j[k] /= 8;
#endif

    // Convert CSR to sell-c-sigma
    sellcs_create_matrix_from_CSR_rd(csr->rows, csr->columns, csr->i,
        (const index_t*)csr->j, csr->A, 0, 0, sellcs_matrix);

#if defined(EPI) && defined(INDEX64)
    for (int k = 0; k < csr->nnz; k++) csr->j[k] *= 8;
#endif

    // Auto-tune execution parameters
    sellcs_analyze_matrix(sellcs_matrix, 0);

    return sellcs_matrix;
}

void mult_sell(void *sellcs_matrix, double *x, double *y)
{
#ifdef EPI
    sellcs_execute_mv_d((sellcs_matrix_t *)sellcs_matrix, x, y);
#else
    sellcs_execute_mv_d_autovector((sellcs_matrix_t *)sellcs_matrix, x, y);
#endif
}
