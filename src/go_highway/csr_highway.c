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
    int64_t *i;
    int64_t *j;
    double *A;
};



void mult_csr_highway(struct csr *csr, double *x, double *y) {
    using T = double;
    
    int nrows = csr->rows;
    //int ncols = csr->columns; 
    int64_t* row_indices =csr->i;
   
    int64_t *col_indices = csr->j;
    double *values = csr->A;

    #ifdef RVV1_M2_256
    const hn::ScalableTag<T, 1> d;
    #else
    const hn::ScalableTag<T> d;
    #endif

    const size_t LANES = hn::Lanes(d);

    const hn::Rebind<int64_t, decltype(d)> di64;

 
    for (int64_t i = 0; i < nrows; i++) {
        const int64_t start_index = row_indices[i];
        const int64_t end_index = row_indices[i + 1];
        int64_t elements = end_index - start_index;
        
        if (elements == 0) {
            y[i] = 0.0;  
            continue;
        }
      
        auto prod_v = hn::Zero(d);
        
        
        // BUCLE PRINCIPAL
        for (int64_t j = 0; j < elements; ) {
            int64_t restantes = elements - j;
            
                // Cargar valores de la matriz
                auto a_vals = hn::LoadN(d, values + start_index + j, restantes);
               
                // Cargar índices de columna (32-bit)
                auto v_idx = hn::LoadN(di64, 
                    reinterpret_cast<const int64_t*>(col_indices + start_index + j), restantes);
              
             
                // Gather desde el vector x
                auto x_vals = hn::GatherIndex(d, x, v_idx);
               
                prod_v = hn::MulAdd(a_vals, x_vals, prod_v);
                
                j += LANES; 

        }

        y[i] = hn::ReduceSum(d, prod_v) ;
    }
  }

} // extern "C"
