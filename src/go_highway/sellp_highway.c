
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
    int *block_ptr;
    int *j;
    double *A;
    size_t memusage;
};



void mult_sellp_highway(struct sellp *sellp, double *x, double *y) {
   
    using T = double;
    uint32_t *col_idx  = (uint32_t *)sellp->j;  
    const hn::ScalableTag<T> d;
    //const size_t LANES = hn::Lanes(d);
    
    const hn::Rebind<int32_t, decltype(d)> di32;
    const hn::Rebind<int64_t, decltype(d)> di64;


    for (int b = 0; b < sellp->num_blocks; b++) {
        auto prod_v = hn::Zero(d);
        int restantes = 0;
        for (int l = sellp->block_ptr[b]; l < sellp->block_ptr[b + 1]; l += _BLOCK) {
           restantes = sellp->block_ptr[b + 1] - l;
           const auto a_vals = hn::LoadN(d, sellp->A + l , restantes);
           auto v_idx_32 = hn::LoadN(di32, reinterpret_cast<int32_t*>(col_idx + l), restantes);

           auto v_idx_64 = hn::PromoteTo(di64, v_idx_32);
           auto x_vals = hn::GatherIndex(d, x, v_idx_64);
           prod_v = hn::MulAdd(a_vals, x_vals, prod_v);
        }
      
        hn::StoreN(prod_v, d, &y[b * _BLOCK],restantes) ;
    }
} 

} // extern "C" 

