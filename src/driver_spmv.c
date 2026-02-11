#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <ctype.h>

#include "spmv.h"
#include "sellp.h"
#include "acsr.h"
#include "colors.h"

#ifdef GO_HIGHWAY
#include "go_highway/csr_highway.h"
#include "go_highway/sellp_highway.h"
#endif

#define MAX_PATH_LEN 1024

double vec_norm(int n, double *x)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

//Return the available memory in Mb
unsigned long available_memory() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;

    char linea[256];
    unsigned long memoria_kb = 0;

    while (fgets(linea, sizeof(linea), f)) {
        if (strncmp(linea, "MemAvailable:", 13) == 0) {
            char *p = linea + 13;
            while (*p && !isdigit(*p)) p++;
            memoria_kb = strtoul(p, NULL, 10);
            break;
        }
    }

    fclose(f);

    return memoria_kb;
}

int enought_memory(const char *fpath, int *orows, int *ocols, int *onnz, unsigned long *onecessary_mem) {
    FILE *f = fopen(fpath, "r");
    if (f == NULL) {
        perror("Error al abrir el fichero");
        return 0;
    }

    char line[1024];
    long rows, cols, nnz;
    int is_sparse = 0;

    // 1. Dense or coordinate?
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "coordinate")) {
            is_sparse = 1;
        }
    }

    // 2. Skip commnets
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // 3. Parse matrix characteristics
    if (is_sparse) {
        sscanf(line, "%ld %ld %ld", &rows, &cols, &nnz);
    } else {
        sscanf(line, "%ld %ld", &rows, &cols);
        nnz = rows * cols;
    }
    fclose(f);

    // 4. Check memory space: We consider doubles values (8 bytes) and Index of 4 bytes
    unsigned long necessary_mem;
    if (is_sparse) {
        // Format COO: row index(4) + col index(4) + values(8) = 16 bytes per NNZ
        necessary_mem = nnz * (sizeof(int) * 2 + sizeof(double));
    } else {
        // Format denso: only double values
        necessary_mem = (unsigned long)rows * cols * sizeof(double);
    }

    //Y vector + X vector
    necessary_mem = (rows + cols) * sizeof(double) + 2 * necessary_mem;

    unsigned long free_memory = available_memory();

    necessary_mem = necessary_mem / (1024 * 1024);
    free_memory   = free_memory  / (1024);

    // Imprimir para depuración (opcional)
    //printf("Necesaria: %lu MB | Disponible: %lu MB\n", 
            //necessary_mem, free_memory);

    *onecessary_mem = necessary_mem;
    *orows = (int)rows;
    *ocols = (int)cols;
    *onnz  = (int)nnz;
    
    return (necessary_mem < (free_memory * 0.95));
}

int main(int argc, char *argv[])
{
    unsigned long necessary_mem;
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
    printf("  +======================================================================================================================+\n");
    printf("  |%s                                                  I N P U T    D A T A                                                %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    printf("  +======================================================================================================================+\n");
    printf("  | Format           :   %s%-90s%s      |\n", COLOR_BOLDCYAN, argv[1], COLOR_RESET);
    printf("  | Input Matrix List:   %s%-90s%s      |\n", COLOR_BOLDCYAN, argv[2], COLOR_RESET);
    printf("  | Outpt CSV        :   %s%-90s%s      |\n", COLOR_BOLDCYAN, argv[3], COLOR_RESET);
    printf("  +======================================================================================================================+\n\n");

    printf("  +======================================================================================================================+\n");
    printf("  |%s                                          D R I V E R    F O R    S P M V                                             %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    printf("  +======================================================================================================================+\n");
    printf("  |%s  Matrix                             ROWS      COLUMNS     NNZ    | GFlops      T(s)    Mem (Mb) |    ERROR     Test  |%s\n", COLOR_RESET, COLOR_RESET);
    printf("  +======================================================================================================================+\n");

    while (fgets(line, sizeof(line), infd) != NULL) {
        line[strcspn(line, "\n")] = '\0'; 
        char *p = strrchr(line, '.');

	char *name = strrchr(line, '/');
	name++;

	//Check for necessary memory
        if (enought_memory(line, &rows, &columns, &nnz, &necessary_mem)) {

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
        
        
            // build sparse matrix
            void *matrix;
            void (*mult)(void *, double *, double *);
        
            if (strcmp(argv[1], "coo") == 0) {
                matrix = create_coo(rows, columns, nnz, coo);
                mult = (void (*)(void *, double *, double *))mult_coo;
            } else if (strcmp(argv[1], "csr_vec") == 0) {
                matrix = create_csr(rows, columns, nnz, _BLOCK, coo);
                mult = (void (*)(void *, double *, double *))mult_csr;
            } else if (strcmp(argv[1], "csr_base") == 0) {
                matrix = create_csr(rows, columns, nnz, 1, coo);
                mult = (void (*)(void *, double *, double *))mult_csr_base;
            } else if (strcmp(argv[1], "csr_highway") == 0) {
                #ifdef GO_HIGHWAY
                matrix = create_csr(rows, columns, nnz, _BLOCK, coo);
                mult = (void (*)(void *, double *, double *))mult_csr_highway;
                #else
		printf("ERROR: Compiled without suppor for Google Highway.\n");
		exit(-1);
                #endif
            } else if (strcmp(argv[1], "sellp_vec") == 0) {
                struct csr * mat_csr = create_csr(rows, columns, nnz, 1, coo);
                matrix = create_sellp(rows, columns, nnz, mat_csr->i, mat_csr->j, mat_csr->A);
	        free_csr(mat_csr);
                mult   = (void (*)(void *, double *, double *))mult_sellp;
            } else if (strcmp(argv[1], "sellp_highway") == 0) {
                #ifdef GO_HIGHWAY
                struct csr * mat_csr = create_csr(rows, columns, nnz, 1, coo);
                matrix = create_sellp(rows, columns, nnz, mat_csr->i, mat_csr->j, mat_csr->A);
	        free_csr(mat_csr);
                mult   = (void (*)(void *, double *, double *))mult_sellp_highway; 
                #else
		printf("ERROR: SELLP for Google Highway not implemented\n");
		exit(-1);
		#endif
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
            
            double *x     = SPMV_ALLOC(double, columns);
            double *y     = SPMV_ALLOC(double, rows + 64);
            double *y_ref = SPMV_ALLOC(double, rows);
        
            for (int i = 0; i < columns; i++) { 
		    x[i] = ((double) rand()) / RAND_MAX;
		    y[i] = ((double) rand()) / RAND_MAX;
	    }

            double norm = vec_norm(columns, x);
            for (int i = 0; i < columns; i++) x[i] /= norm;
            mult_mtx(rows, columns, nnz, coo, x, y_ref);
        
            mult(matrix, x, y);
      
           double error = 0.0;
           double nrm   = 0.0;

           for (int i = 0; i < rows; i++) {
               double tmp = y_ref[i];
               nrm       += tmp *tmp;
               tmp        = fabs(y[i] - y_ref[i]);
               error     += tmp * tmp;
           }

           if (nrm != 0.0) error = sqrt(error) / sqrt(nrm);
           else            error = sqrt(error);

       
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
    

	    printf("  | %s%-30s%s %10d %10d %10d  | %s%.4f%s  %.4e  %8ld | %.4e  ", COLOR_BOLDCYAN, name, COLOR_RESET, rows, columns, nnz, 
			                                                            COLOR_BOLDYELLOW, GFlops, COLOR_RESET, ftime, necessary_mem, error);
	    if (error < 1E-10) printf("  %sOK%s   |\n", COLOR_BOLDGREEN, COLOR_RESET);	
	    else               printf("  %sERR%s  |\n", COLOR_BOLDRED, COLOR_RESET);	
    
	    fprintf(outfd, "%s;%s;%d;%d;%d;%.5e;%.5e\n", argv[1], name, rows, columns, nnz, GFlops, ftime);
    
            SPMV_FREE(x);
            SPMV_FREE(y); 
            SPMV_FREE(y_ref);
   	    free(coo);
    
            if      (strcmp(argv[1], "coo") == 0)      free_coo((struct coo *)matrix);
            else if (strcmp(argv[1], "csr") == 0)      free_csr((struct csr *)matrix);
            else if (strcmp(argv[1], "csr_base") == 0) free_csr((struct csr *)matrix);
            else if (strcmp(argv[1], "sellp") == 0)    free_sellp((struct sellp *)matrix);
	} else {
	    printf("  | %s%-30s%s %10d %10d %10d  | %s%.4f%s  %.4e  %8ld | %.4e  ", COLOR_BOLDCYAN, name, COLOR_RESET, rows, columns, nnz, 
			                                                             COLOR_BOLDYELLOW, 0.0, COLOR_RESET, 0.0, necessary_mem, 0.0);
	    printf("  %s---%s  |\n", COLOR_BOLDRED, COLOR_RESET);	
	    fprintf(outfd, "%s;%s;%d;%d;%d;%.5e;%.5e\n", argv[1], name, rows, columns, nnz, 0.0, 0.0);
	}
    
    }

    printf("  +======================================================================================================================+\n");
    printf("\n\n");
    fclose(infd);
    fclose(outfd);

    return 0;
}


