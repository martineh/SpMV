#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <cstdint>
#include <iostream>
#include "omp.h"

namespace hn = hwy::HWY_NAMESPACE;

//#define HWY_TARGETS HWY_RVV HWY_NEON HWY_SVE2

extern "C" {


struct csr {
    int rows, columns, nnz;
    long memusage;
    uint32_t *i;
    uint32_t *j;
    double *A;
};



void mult_csr_highway(struct csr *csr, double *x, double *y) {
    using T = double;
    
    int nrows = csr->rows;
    uint32_t *row_index = csr->i;
    uint32_t *col_index = csr->j;
    double *values = csr->A;

    #ifdef RVV1_M2_256
    const hn::ScalableTag<T, 1> d;
    #else
    const hn::ScalableTag<T> d;
    #endif

    const int LANES = hn::Lanes(d);

    const hn::Rebind<int32_t, decltype(d)> di32;
    const hn::Rebind<int64_t, decltype(d)> di64;

    int j;

    for (int i = 0; i < nrows; i++) {
        const uint32_t start_index = row_index[i];
        const uint32_t end_index = row_index[i + 1];
        int elements = end_index - start_index;
        
        auto prod_v = hn::Zero(d);
        
        for (j = 0; j < elements - (LANES - 1); j += LANES) {
            const auto a_vals = hn::Load(d, values + start_index + j);
            auto v_idx_32 = hn::Load(di32, 
                reinterpret_cast<const int32_t*>(col_index + start_index + j));

            auto v_idx_64 = hn::PromoteTo(di64, v_idx_32);
            auto x_vals   = hn::GatherIndex(d, x, v_idx_64);
            
	    prod_v        = hn::MulAdd(a_vals, x_vals, prod_v);
        }
        y[i] = hn::ReduceSum(d, prod_v) ;

	for (;j < elements; j++) y[i] += x[col_index[start_index + j]] * values[start_index + j];

    }
  }

} // extern "C"
