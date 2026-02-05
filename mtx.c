#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "spmv.h"

int by_row_mtx(const struct mtx *a, const struct mtx *b)
{
    if (a->i < b->i) return -1;
    if (a->i > b->i) return 1;
    if (a->j < b->j) return -1;
    if (a->j > b->j) return 1;
    return 0;
}

void sort_mtx(int nnz, struct mtx *coo, int (*compar)(const struct mtx *, const struct mtx *))
{
    bool sorted = true;
    int i = 1;
    while (sorted && i < nnz) {
        // if (coo[i - 1].a > coo[i],a) {
        int c = (*compar)(&coo[i - 1], &coo[i]);
        if (c == 1) {
            sorted = false;
        }
        i++;
    }
    if (!sorted) {
        qsort(coo, nnz, sizeof(struct mtx),
                (int (*)(const void *, const void *))compar);
    }
}

void load_mtx(const char *fname, int *rows, int *columns, int *nnz, struct mtx **a)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(1);
    }

    char header[101],   // %%MatrixMarket
         object[101],   // matrix, vector
         format[101],   // coordinate, array
         type[101],     // real, double, complex, integer, pattern
         symmetry[101]; // general, symmetric, skew-symmetric, hermitian

    if (fscanf(f, "%100s %100s %100s %100s %100s\n",
            header, object, format, type, symmetry) != 5) {
        fprintf(stderr, "Error reading header from file: %s\n", fname);
        exit(1);
    }

    if (strcmp(header, "%%MatrixMarket")) {
        fprintf(stderr, "Missing header in file: %s\n", fname);
        exit(1);
    }

    if (strcmp(object, "matrix") || strcmp(format, "coordinate")) {
        fprintf(stderr, "Object (%s) and/or format (%s) not supported in file: %s\n",
                object, format, fname);
        exit(1);
    }

    // skip over comments
    char buffer[1000];
    do {
        if (fgets(buffer, sizeof(buffer), f) == NULL) {
            fprintf(stderr, "Error reading matrix size from file: %s\n", fname);
            exit(1);
        }
    } while (buffer[0] == '%');

    // read matrix size
    long m, n, nz;
    if (sscanf(buffer, "%ld %ld %ld\n", &m, &n, &nz) != 3) {
        fprintf(stderr, "Error reading matrix size (%s) from file: %s\n",
            buffer, fname);
        exit(1);
    }
    if (m > INT_MAX || n > INT_MAX || nz > INT_MAX) {
        fprintf(stderr, "Matrix too big (%s): %s\n", buffer, fname);
        exit(1);
    }

    bool symmetric = strcmp(symmetry, "general") != 0;
    struct mtx *coo;
    if (symmetric) {
        coo = (struct mtx*)malloc(sizeof(struct mtx) * 2 * nz);
    } else {
        coo = (struct mtx*)malloc(sizeof(struct mtx) * nz);
    }
    if (coo == NULL) {
        fprintf(stderr, "Out of memory loading: %s\n", fname);
        exit(1);
    }

    if (!strcmp(type, "complex")) {
        fprintf(stderr, "Matrix is complex (%s): %s\n", type, fname);
        exit(1);
    }
    bool pattern = strcmp(type, "pattern") == 0;
    int k = 0;
    double real = 1.0;
    for (int l = 0; l < nz; l++) {
        if (pattern) {
            if (fscanf(f, "%d %d\n", &coo[k].i, &coo[k].j) != 2) {
                fprintf(stderr, "Error reading matrix element %i\n", l);
                exit(1);
            }
        } else {
            if (fscanf(f, "%d %d %le\n", &coo[k].i, &coo[k].j, &real) != 3) {
                fprintf(stderr, "Error reading matrix element %i'\n", l);
                exit(1);
            }
            if (real == 0.0) continue;
        }
        if (coo[k].i < 1 || coo[k].i > m || coo[k].j < 1 || coo[k].j > n) {
            fprintf(stderr, "Coordinates (%i,%i) out of range in line %i\n", coo[k].i, coo[k].j, l);
            exit(1);
        }
        coo[k].i--;
        coo[k].j--;
        coo[k].a = real;
        if (symmetric && coo[k].i != coo[k].j) {
            coo[k + 1].i = coo[k].j;
            coo[k + 1].j = coo[k].i;
            coo[k + 1].a = coo[k].a;
            k++;
        }
        k++;
    }
    fclose(f);
    *rows = m; *columns = n; *nnz = k; *a = coo;
}

void load_bin(const char *fname, int *rows, int *columns, int *nnz, struct mtx **a)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(1);
    }

    // read matrix size
    int m, n;
    long nz;
    if (fread(&m, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error reading file: %s\n", fname);
        exit(1);
    }
    if (fread(&n, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error reading file: %s\n", fname);
        exit(1);
    }
    if (fread(&nz, sizeof(long), 1, f) != 1) {
        fprintf(stderr, "Error reading file: %s\n", fname);
        exit(1);
    }
    if (nz > INT_MAX) {
        fprintf(stderr, "Matrix too big (%ld): %s\n", nz, fname);
        exit(1);
    }

    // read matrix elements
    struct mtx *coo = (struct mtx *)malloc(sizeof(struct mtx) * nz);
    if (coo == NULL) {
        fprintf(stderr, "Out of memory loading: %s\n", fname);
        exit(1);
    }

    if (sizeof(struct mtx) == 16) {
        // the struct is not padded so we can read it in one fread
        if (fread(coo, sizeof(struct mtx), nz, f) != nz) {
            fprintf(stderr, "Error reading file: %s\n", fname);
            exit(1);
        }
    } else {
        for (int k = 0; k < nz; k++) {
            int i, j;
            double x;
            if (fread(&i, sizeof(int), 1, f) != 1) {
                fprintf(stderr, "Error reading file: %s\n", fname);
                exit(1);
            }
            if (fread(&j, sizeof(int), 1, f) != 1) {
                fprintf(stderr, "Error reading file: %s\n", fname);
                exit(1);
            }
            if (fread(&x, sizeof(double), 1, f) != 1) {
                fprintf(stderr, "Error reading file: %s\n", fname);
                exit(1);
            }

            coo[k].i = i;
            coo[k].j = j;
            coo[k].a = x;
        }
    }

    fclose(f);
    *rows = m; *columns = n; *nnz = nz; *a = coo;
}

void save_bin(const char *fname, int rows, int columns, int nnz, struct mtx *coo)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(1);
    }

    // write matrix size
    long nz = nnz;
    if (fwrite(&rows, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error writing file: %s\n", fname);
        exit(1);
    }
    if (fwrite(&columns, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error writing file: %s\n", fname);
        exit(1);
    }
    if (fwrite(&nz, sizeof(long), 1, f) != 1) {
        fprintf(stderr, "Error writing file: %s\n", fname);
        exit(1);
    }

    // write matrix elements
    for (int k = 0; k < nz; k++) {
        if (fwrite(&coo[k].i, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error writing file: %s\n", fname);
            exit(1);
        }
        if (fwrite(&coo[k].j, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error writing file: %s\n", fname);
            exit(1);
        }
        if (fwrite(&coo[k].a, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "Error writing file: %s\n", fname);
            exit(1);
        }
    }

    fclose(f);
}

void print_mtx(int nnz, struct mtx *coo)
{
    for (int k = 0; k < nnz; k++)
        printf("%d %d %e\n", coo[k].i, coo[k].j, coo[k].a);
}

void spy_mtx(int rows, int columns, int nnz, struct mtx *coo)
{
    char *s = SPMV_ALLOC(char, columns + 2);
    s[columns] = '\n';
    s[columns + 1] = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) s[j] = ' ';
        for (int k = 0; k < nnz; k++) {
            if (coo[k].i == i) s[coo[k].j] = '.';
        }
        puts(s);
    }
    free(s);
}

void mult_mtx(int rows, int columns, int nnz, struct mtx *coo, double *x, double *y)
{
    for (int i = 0; i < rows; i++) y[i] = 0.0;
    for (int k = 0; k < nnz; k++) {
        y[coo[k].i] += x[coo[k].j] * coo[k].a;
    }
}

void create_band(int n, int width, int *nnz, struct mtx **coo)
{
    int nz = 0;
    for (int i = 0; i < n; i++) {
        int j = i - width + 1;
        if (j < 0) j = 0;
        int m = i + width;
        if (m > n) m = n;
        for (; j < m; j++) nz++;
    }
    struct mtx *a = SPMV_ALLOC(struct mtx, nz);

    int k = 0;
    for (int i = 0; i < n; i++) {
        int j = i - width + 1;
        if (j < 0) j = 0;
        int m = i + width;
        if (m > n) m = n;
        for (; j < m; j++) {
            a[k].a = (double)rand() / RAND_MAX;
            a[k].i = i;
            a[k].j = j;
            k++;
        }
    }
    *nnz = nz;
    *coo = a;
}

void create_arrow(int n, int width, int *nnz, struct mtx **coo)
{
    int nz = 0;
    for (int i = 0; i < width; i++)
        for (int j = 0; j < n; j++)
            nz++;
    for (int i = width; i < n; i++) {
        for (int j = 0; j < width; j++) nz++;
        int j = i - width + 1;
        if (j < width) j = width;
        int m = i + width;
        if (m > n) m = n;
        for (; j < m; j++) nz++;
    }
    struct mtx *a = SPMV_ALLOC(struct mtx, nz);

    int k = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < n; j++) {
            a[k].a = (double)rand() / RAND_MAX;
            a[k].i = i;
            a[k].j = j;
            k++;
        }
    }
    for (int i = width; i < n; i++) {
        for (int j = 0; j < width; j++) {
            a[k].a = (double)rand() / RAND_MAX;
            a[k].i = i;
            a[k].j = j;
            k++;
        }
        int j = i - width + 1;
        if (j < width) j = width;
        int m = i + width;
        if (m > n) m = n;
        for (; j < m; j++) {
            a[k].a = (double)rand() / RAND_MAX;
            a[k].i = i;
            a[k].j = j;
            k++;
        }
    }
    *nnz = nz;
    *coo = a;
}
