#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "spmv.h"
#include "sellp.h"
#include "acsr.h"

#include "colors.h"

#define MAX_PATH_LEN 1024

double vec_norm(int n, double *x)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

int main(int argc, char *argv[])
{
    int rows, columns, nnz;
    struct mtx *coo;

    // get parallel configuration
    //int np = 1;
    //int max_threads = omp_get_max_threads();

    FILE *infd, *outfd;
    char line[MAX_PATH_LEN];

    // create matrix or load from file
    if (argc != 4) {
        printf("Usage: spmv_power [coo|csr|csr_base|sellp] <mtx list> <outpt>\n");
        exit(-1);
    }

    infd  = fopen(argv[2], "r");
    outfd = fopen(argv[3], "w");

    if (infd == NULL) {
        perror("Error openning file input for matrix list.\n");
        exit(-1);
    }
	
    fprintf(outfd, "Format;File;Rows;Columns;NNZ;GFlops;Time(s)\n");


    printf("\n\n");
    printf("  +============================================================================================================+\n");
    printf("  |%s                                             I N P U T    D A T A                                           %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    printf("  +============================================================================================================+\n");
    printf("  | Format           :   %s%-80s%s      |\n", COLOR_BOLDCYAN, argv[1], COLOR_RESET);
    printf("  | Input Matrix List:   %s%-80s%s      |\n", COLOR_BOLDCYAN, argv[2], COLOR_RESET);
    printf("  | Outpt CSV        :   %s%-80s%s      |\n", COLOR_BOLDCYAN, argv[3], COLOR_RESET);
    printf("  +============================================================================================================+\n\n");

    printf("  +============================================================================================================+\n");
    printf("  |%s                                     D R I V E R    F O R    S P M V                                        %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    printf("  +============================================================================================================+\n");
    printf("  |%s  Matrix                             ROWS      COLUMNS     NNZ    | GFlops      T(s)   |    ERROR     Test  |%s\n", COLOR_RESET, COLOR_RESET);
    printf("  +============================================================================================================+\n");

    while (fgets(line, sizeof(line), infd) != NULL) {
        line[strcspn(line, "\n")] = '\0'; 
        char *p = strrchr(line, '.');

        if (p != NULL && strcmp(p, ".bin") == 0) {
            load_bin(line, &rows, &columns, &nnz, &coo);
        } else if (p != NULL && strcmp(p, ".mtx") == 0) {
            load_mtx(line, &rows, &columns, &nnz, &coo);
        } else {
            printf("Unkown file extension\n");
            return 1;
        }
    
        if (rows != columns) {
            printf("Matrix is not square\n");
            return 1;
        }
    
        // test result part 1
        double *x     = SPMV_ALLOC(double, columns);
        double *y     = SPMV_ALLOC(double, rows);
        double *y_ref = SPMV_ALLOC(double, rows);
    
        for (int i = 0; i < columns; i++) x[i] = y[i] = 1.0;
    
        double norm = vec_norm(columns, x);
        for (int i = 0; i < columns; i++) x[i] /= norm;
        mult_mtx(rows, columns, nnz, coo, x, y_ref);
    
        // build sparse matrix
        void *matrix;
        void (*mult)(void *, double *, double *);
    
        if (strcmp(argv[1], "coo") == 0) {
            matrix = create_coo(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_coo;
        } else if (strcmp(argv[1], "csr") == 0) {
            matrix = create_csr_pad(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr;
        } else if (strcmp(argv[1], "csr_base") == 0) {
            matrix = create_csr_pad(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_base;
        } else if (strcmp(argv[1], "acsr") == 0) {
            struct csr * mat_csr = create_csr_pad(rows, columns, nnz, coo);
	    matrix = create_acsr(rows, columns, nnz, mat_csr->A, mat_csr->j, mat_csr->i);
            mult = (void (*)(void *, double *, double *))mult_acsr;
        } else if (strcmp(argv[1], "csr_merge") == 0) {
            matrix = create_csr_merge(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_merge;
        } else if (strcmp(argv[1], "csr_bal") == 0) {
            matrix = create_csr_bal(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_bal;
        } else if (strcmp(argv[1], "ell") == 0) {
            matrix = create_ell(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_ell;
        } else if (strcmp(argv[1], "sellp") == 0) {
            struct csr * mat_csr = create_csr_pad(rows, columns, nnz, coo);
            matrix = create_sellp(rows, columns, nnz, mat_csr->i, mat_csr->j, mat_csr->A);
            mult   = (void (*)(void *, double *, double *))mult_sellp;
	    free_csr(mat_csr);
        } else if (strcmp(argv[1], "sell") == 0) {
            #ifdef INDEX64
            struct csr_epi *csr_matrix = create_csr_epi(rows, columns, nnz, coo);
            #else
            struct csr *csr_matrix = create_csr(rows, columns, nnz, 1, coo);
            #endif
            free(coo); // free some memory before building SELL
            matrix = create_sell(csr_matrix);
            mult = mult_sell;
        #ifdef MKL
        } else if (strcmp(argv[1], "csr_mkl") == 0) {
            matrix = create_csr_mkl(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_csr_mkl;
        #endif
        #ifdef PETSC
        } else if (strcmp(argv[1], "petsc") == 0) {
            matrix = create_petsc(rows, columns, nnz, coo);
            mult = (void (*)(void *, double *, double *))mult_petsc;
        #endif
        } else {
            printf("Unknown matrix format\n");
            return 1;
        }
    
        mult(matrix, x, y);
    
        double res = 0.0;
        for (int i = 0; i < rows; i++) {
            double d = fabs(y[i] - y_ref[i]);
            // if (d > 1e-5) printf("diff %d %e %e\n", i, y[i], y_ref[i]);
            if (d > res) res = d;
        }
    
        double time_spmv = 0.0;
        double t0 = get_time();
        int it = 0;
        for (;;) {
            it++;
            double t1 = get_time();
            mult(matrix, x, y);
            double t2 = get_time();
            time_spmv += t2 - t1;
            norm = vec_norm(columns, y);
            for (int i = 0; i < columns; i++) x[i] = y[i] / norm;
            int stop = t2 > 2 + t0 && it >= 10;
            if (stop) break;
        }
    

	double GFlops = 2.0 * nnz * it / time_spmv / 1000000000;
	double ftime  = time_spmv / it;

	char *name = strrchr(line, '/');
	name++;

	printf("  | %s%-30s%s %10d %10d %10d  | %s%.4f%s  %.4e | %.4e  ", COLOR_BOLDCYAN, name, COLOR_RESET, rows, columns, nnz, COLOR_BOLDYELLOW, GFlops, COLOR_RESET, ftime, res);
   
	if (res < 1E-8) printf("  %sOK%s   |\n", COLOR_BOLDGREEN, COLOR_RESET);	
	else            printf("  %sERR%s  |\n", COLOR_BOLDRED, COLOR_RESET);	

	fprintf(outfd, "%s;%s;%d;%d;%d;%.5e;%.5e\n", argv[1], name, rows, columns, nnz, GFlops, ftime);

        SPMV_FREE(x);
        SPMV_FREE(y); 
        SPMV_FREE(y_ref);
    
        if      (strcmp(argv[1], "coo") == 0)      free_coo((struct coo *)matrix);
        else if (strcmp(argv[1], "csr") == 0)      free_csr((struct csr *)matrix);
        else if (strcmp(argv[1], "csr_base") == 0) free_csr((struct csr *)matrix);
        else if (strcmp(argv[1], "sellp") == 0)    free_sellp((struct sellp *)matrix);
    
    }

    printf("  +============================================================================================================+\n");
    printf("\n\n");
    fclose(infd);
    fclose(outfd);

    return 0;
}


