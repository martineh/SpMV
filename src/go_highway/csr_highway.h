#ifndef CSR_HIGHWAY_H
#define CSR_HIGHWAY_H

#ifdef __cplusplus
extern "C" {
#endif

struct csr;  


void mult_csr_highway(struct csr *csr, double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
