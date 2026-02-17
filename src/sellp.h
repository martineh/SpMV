

struct sellp {
    int rows;
    int columns;
    int nnz;
    int num_blocks;
    int *block_ptr;
    int64_t *j;
    double *A;
    size_t memusage;
};


struct sellp *create_sellp(int rows, int columns, int nnz, const int64_t *row_ptr_csr, const int64_t *col_idx, const double *A_csr);
void mult_sellp(struct sellp *sellp, double *x, double *y);
void mult_sellp_autovec(struct sellp *ell, double *x, double *y);
void free_sellp(struct sellp *sellp);

