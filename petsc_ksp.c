#include <petsc.h>
#include <libgen.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "spmv.h"

int main(int argc, char *argv[])
{
    int rows, columns, nnz;
    struct mtx *coo;

    // get parallel configuration
    int max_threads = 1;
    int np = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    MPI_Comm_size(PETSC_COMM_WORLD, &np);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // create matrix or load from file
    char mtx_file[PATH_MAX];
    PetscBool set;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mtx", mtx_file, sizeof(mtx_file), &set));
    if (!set) {
        printf("MTX file missing (use -mtx flag)\n");
        goto petsc_finalize;
    } else {
        load_mtx(mtx_file, &rows, &columns, &nnz, &coo);
        if (rows != columns) {
            printf("Matrix is not square\n");
            goto petsc_finalize;
        }
    }

    // build sparse matrix
    Mat A;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, rows, columns));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    assembly_petsc_mat(A, rows, columns, nnz, coo);
    const char *type;
    PetscCall(MatGetType(A, &type));

    Vec x, u, b;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, rows));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(VecSet(u, 1.0));
    PetscCall(VecDuplicate(x, &b));
    PetscCall(MatMult(A, u, b));

    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    double t1 = get_time();
    PetscCall(KSPSolve(ksp, b, x));
    double t2 = get_time();
    PetscCall(VecAXPY(x, -1.0, u));
    PetscScalar res;
    PetscCall(VecNorm(x, NORM_2, &res));
    PetscInt its;
    PetscCall(KSPGetIterationNumber(ksp, &its));

    char *filename = basename(mtx_file);
    char *extension = strrchr(filename, '.');
    if (extension) *extension = 0;
    if (rank == 0) {
        printf("%s %s %i %i %i %i %i %e %i %e\n",
            type, filename, max_threads, np,
            rows, columns, nnz,
            (t2 - t1) / its,
            (int)its, res);
    }

    petsc_finalize:
    PetscCall(PetscFinalize());
    return 0;
}
