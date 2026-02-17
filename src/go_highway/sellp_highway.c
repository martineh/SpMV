
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <cstdint>
#include <iostream>
#include "omp.h"
#include "../spmv.h"

namespace hn = hwy::HWY_NAMESPACE;

extern "C" {

struct sellp {
    int rows;
    int columns;
    int nnz;
    int num_blocks;
    int64_t *block_ptr;
    int64_t  *j;
    double *A;
    size_t memusage;
};



void mult_sellp_highway(struct sellp *sellp, double *x, double *y) {
   
    using T = double;
    int64_t* col_idx = sellp->j;
    int64_t* block_ptr = sellp->block_ptr;
    int num_blocks = sellp->num_blocks;

    #ifdef RVV1_M2_256
    const hn::ScalableTag<T, 1> d;
    #else
    const hn::ScalableTag<T> d;
    #endif
    
    //const size_t LANES = hn::Lanes(d);
   
    const hn::Rebind<int64_t,decltype(d)> di64;
   
    for (int64_t b = 0; b < num_blocks  ; b++) {
        auto prod_v = hn::Zero(d);
        int restantes = 0;
        for (int64_t l = block_ptr[b]; l < block_ptr[b + 1]; l += _BLOCK) {
           restantes = block_ptr[b + 1] - l;
           const auto a_vals = hn::LoadN(d, sellp->A + l , restantes);
           auto v_idx = hn::LoadN(di64, col_idx + l,restantes);

           auto x_vals = hn::GatherIndex(d, x, v_idx);
           prod_v = hn::MulAdd(a_vals, x_vals, prod_v);
        }
      
        hn::StoreN(prod_v, d, &y[b * _BLOCK],restantes) ;
    }
} 

} // extern "C" 

