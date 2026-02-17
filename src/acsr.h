
struct acsr {
    double *v_values;
    int64_t    *v_columns;
    int64_t    *v_rowptr;
    int    num_vectors;
    int    rows, columns, nnz;
};


struct acsr *create_acsr(int rows, int ncols, int nnz, double *values, int64_t   *columns, int64_t *row_ptr);
void mult_acsr(struct acsr *acsr, double *x, double *y);

