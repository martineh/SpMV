#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cstdint>

#include <omp.h>

#ifdef SVE_128
  #define _BLOCK 2
  #define ALIGN_BYTES 64
#elif SVE_256
  #define _BLOCK 4
  #define ALIGN_BYTES 128
#elif NEON
  #define _BLOCK 2
  #define ALIGN_BYTES 32
#elif AVX2
  #define _BLOCK 4
  #define ALIGN_BYTES 32
#endif


struct mtx { int i; int j; double a; };

void load_mtx(const char *fname, int *rows, int *columns, int *nnz, struct mtx **coo);
void load_bin(const char *fname, int *rows, int *columns, int *nnz, struct mtx **coo);
void save_bin(const char *fname, int rows, int columns, int nnz, struct mtx *coo);
void create_band(int rows, int width, int *nnz, struct mtx **coo);
void create_arrow(int rows, int width, int *nnz, struct mtx **coo);
int by_row_mtx(const struct mtx *a, const struct mtx *b);
void sort_mtx(int nnz, struct mtx *coo, int (*compar)(const struct mtx *, const struct mtx *));
void mult_mtx(int rows, int columns, int nnz, struct mtx *coo, double *x, double *y);
void spy_mtx(int rows, int columns, int nnz, struct mtx *coo);
void print_mtx(int nnz, struct mtx *coo);

// CSR matrix

struct csr {
    int rows, columns, nnz;
    long memusage;
    int *i;
    int *j;
    double *A;
};

struct csr *create_csr_pad(int rows, int columns, int nnz, struct mtx *mtx);
struct csr *create_csr(int rows, int columns, int nnz, int alignment, struct mtx *mtx);
void free_csr(struct csr *csr);
void mult_csr(struct csr *csr, double *x, double *y);
void mult_csr_base(struct csr *csr, double *x, double *y);
void mult_mv_csr(struct csr *csr, int n, double *x, double *y);

struct csr *create_csr_numa(int rows, int columns, int nnz, struct mtx *coo);
void mult_csr_numa(struct csr *csr, double *x, double *y);

struct csr_epi {
    int rows, columns, nnz;
    long memusage;
    long *i;
    long *j;
    double *A;
};

struct csr_epi *create_csr_epi(int rows, int columns, int nnz, struct mtx *coo);
void mult_csr_epi(struct csr_epi *csr, double *x, double *y);

struct csri {
    int rows, columns, nnz;
    long memusage;
    int nb;
    int *block_ptr;
    int *block_size;
    int *block_width;
    int *block_len;
    int *row_id;
    int *j;
    double *A;
};

struct csri *create_csri(int rows, int columns, int nnz, struct mtx *mtx);
void mult_csri(struct csri *csr, double *x, double *y);

struct csr_merge {
    int rows, columns, nnz;
    long memusage;
    int *row_ptr;
    int *col_ind;
    double *val;
    int nthreads;
    int *row_carry;
    double *value_carry;
};

struct csr_merge *create_csr_merge(int rows, int columns, int nnz, struct mtx *coo);
void mult_csr_merge(struct csr_merge *csr, double *x, double *y);

    struct csr_bal_thread {
        int rows;
        int first_row;
        int last_row;
        int *i;
        int *j;
        double *A;
    } ;


struct csr_bal {
    int rows, columns, nnz;
    long memusage;
    int threads;
    struct csr_bal_thread p[];
};

struct csr_bal *create_csr_bal(int rows, int columns, int nnz, struct mtx *coo);
void mult_csr_bal(struct csr_bal *csr, double *x, double *y);

// ELL (ELLPACK with padding)


// ELL (ELLPACK with padding)

struct ell {
    int rows, columns, nnz;
    long memusage;
    int num_blocks;
    int *block_ptr;
    int64_t *j;
    double *A;
};

struct ell *create_ell(int rows, int columns, int nnz, struct mtx *mtx);
void mult_ell(struct ell *ell, double *x, double *y);

// ELL (ELLPACK with padding and sorting)

struct ell_sort {
    int rows, columns, nnz;
    long memusage;
    int num_blocks;
    int *block_ptr;
    int *j;
    double *A;
    int *perm;
};

struct ell_sort *create_ell_sort(int rows, int columns, int nnz, struct mtx *mtx);
void mult_ell_sort(struct ell_sort *ell, double *x, double *y);




// ELL0 (ELLPACK without padding)

    struct ell0_thread {
        int rows;
        int shared_row;
        int nb;
        int *block_ptr;
        int *block_width;
        int *block_len;
        int *row_id;
        int *j;
        double *A;
    } ;


struct ell0 {
    int rows, columns, nnz;
    long memusage;
    int threads;
    struct ell0_thread p[];
};

struct ell0 *create_ell0(int rows, int columns, int nnz, struct mtx *mtx);
void mult_ell0(struct ell0 *ell, double *x, double *y);

// COO matrix

struct coo {
    int rows, columns, nnz;
    long memusage;
    int *i;
    int *j;
    double *A;
};

struct coo *create_coo(int rows, int columns, int nnz, struct mtx *coo);
void free_coo(struct coo *coo);
void mult_coo(struct coo *coo, double *x, double *y);

// PCSR matrix

struct pcsr {
    int rows, columns, nnz;
    long memusage;
    int num_rows;
    int *i;             // row
    int *row_start;     // first element of row
    int *j1;            // column (block)
    unsigned short *j2; // column (element)
    double *A;          // elements
};

struct pcsr *create_pcsr(int rows, int columns, int nnz, struct mtx *pcsr, int block_size);
void mult_pcsr(struct pcsr *pcsr, double *x, double *y);


struct jds {
    int rows, columns, nnz;
    long memusage;
    int num_diags;
    int num_row1;
    int num_blocks;
    int *num_diag;
    int *jd_ptr;
    int *perm;
    int *j;
    double *A;
};

struct jds *create_jds(int rows, int columns, int nnz, struct mtx *mtx);
void mult_jds(struct jds *jds, double *x, double *y);


// SELL format (BSC)
#ifdef INDEX64
void *create_sell(struct csr_epi * csr);
#else
void *create_sell(struct csr * csr);
#endif
void mult_sell(void *sellcs_matrix, double *x, double *y);

#ifdef MKL
#include <mkl.h>

struct csr_mkl {
    int rows, columns, nnz;
    long memusage;
    int *i;
    int *j;
    double *A;
    sparse_matrix_t matrix;
    struct matrix_descr descr;
};

struct csr_mkl *create_csr_mkl(int rows, int columns, int nnz, struct mtx *coo);
void mult_csr_mkl(struct csr_mkl *csr, double *x, double *y);

#endif

#ifdef PETSC
#include <petscmat.h>

struct petsc_mat {
    Mat A;
    Vec x;
    Vec y;
};

struct petsc_mat *create_petsc(int rows, int columns, int nnz, struct mtx *coo);
PetscErrorCode assembly_petsc_mat(Mat A, int rows, int columns, int nnz, struct mtx *mtx);
void mult_petsc(struct petsc_mat *mat, double *x, double *y);

#endif

// Utility functions

#define SPMV_ALLOC(t, l) (t*)spmv_alloc(sizeof(t), l, __FILE__, __func__, __LINE__)
static inline void *spmv_alloc(size_t t, size_t l, const char *file, const char *func, int line)
{
    if (l == 0) return NULL;
    void *p = aligned_alloc(64, t * l); // for AVX512
    if (p == NULL) {
        fprintf(stderr, "Out of memory in %s (%s:%d) allocating %zd elements of size %zd\n",
            func, file, line, l, t);
        abort();
    }
    return p;
}

#define SPMV_FREE(t) free(t)

static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
