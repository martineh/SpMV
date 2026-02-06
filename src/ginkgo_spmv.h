
struct mtx_coo { int i; int j; double a; };

struct sellcs {
    int rows, columns, nnz;
    int num_blocks;
    int *block_ptr;
    int64_t *j;
    double *A;
    long memusage;
};

struct sellcs *create_sellcs(int rows, int columns, int nnz, int *i, int *j, double *A);
void mult_sellcs(struct sellcs *sellcs, double *x, double *y);

