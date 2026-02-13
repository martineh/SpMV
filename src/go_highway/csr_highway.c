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
    //int ncols = csr->columns; 
    uint32_t *row_indices = csr->i;
    uint32_t *col_indices = csr->j;
    double *values = csr->A;

    #ifdef RVV1_M2_256
    const hn::ScalableTag<T, 1> d;
    #else
    const hn::ScalableTag<T> d;
    #endif

    const size_t LANES = hn::Lanes(d);

    const hn::Rebind<int32_t, decltype(d)> di32;
    const hn::Rebind<int64_t, decltype(d)> di64;

    # pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
        const uint32_t start_index = row_indices[i];
        const uint32_t end_index = row_indices[i + 1];
        int elements = end_index - start_index;
        
        if (elements == 0) {
            y[i] = 0.0;  
            continue;
        }

        auto prod_v = hn::Zero(d);
        
        
        // BUCLE PRINCIPAL
        for (int j = 0; j < elements; ) {
            int restantes = elements - j;
                // Cargar valores de la matriz
                const auto a_vals = hn::LoadN(d, values + start_index + j, restantes);
                
                // Cargar índices de columna (32-bit)
                auto v_idx_32 = hn::LoadN(di32, 
                    reinterpret_cast<const int32_t*>(col_indices + start_index + j), restantes);
                
                // Promover a 64-bit para gather
                auto v_idx_64 = hn::PromoteTo(di64, v_idx_32);
                
                // Gather desde el vector x
                auto x_vals = hn::GatherIndex(d, x, v_idx_64);
                
                prod_v = hn::MulAdd(a_vals, x_vals, prod_v);
                
                j += LANES; 

        }

        y[i] = hn::ReduceSum(d, prod_v) ;
    }
  }

} // extern "C"
