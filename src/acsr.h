
struct acsr {
    double *v_values;
    int    *v_columns;
    int    *v_rowptr;
    int    num_vectors;
    int    rows, columns, nnz;
};


struct acsr *create_acsr(int rows, int ncols, int nnz, double *values, int *columns, int *row_ptr);
void mult_acsr(struct acsr *acsr, double *x, double *y);

