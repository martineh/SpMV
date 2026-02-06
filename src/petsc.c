#include "spmv.h"

#ifdef PETSC

PetscErrorCode mult_petsc_private(struct petsc_mat *petsc, double *x, double *y)
{
    PetscFunctionBeginUser;
    PetscCall(VecPlaceArray(petsc->x, x));
    PetscCall(VecPlaceArray(petsc->y, y));
    PetscCall(MatMult(petsc->A, petsc->x, petsc->y));
    PetscCall(VecResetArray(petsc->x));
    PetscCall(VecResetArray(petsc->y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

void mult_petsc(struct petsc_mat *petsc, double *x, double *y)
{
    mult_petsc_private(petsc, x, y);
}

PetscErrorCode assembly_petsc_mat(Mat A, int rows, int columns, int nnz, struct mtx *mtx)
{
    PetscFunctionBeginUser;
    PetscInt start, end;
    PetscCall(MatGetOwnershipRange(A, &start, &end));
    PetscInt *d_nnz = SPMV_ALLOC(PetscInt, end - start);
    PetscInt *o_nnz = SPMV_ALLOC(PetscInt, end - start);
    for (int k = 0; k < end - start; k++) {
        d_nnz[k] = 0;
        o_nnz[k] = 0;
    }
    const char* type;
    PetscCall(MatGetType(A, &type));
    if (strcmp(type, MATSEQSBAIJ) == 0 || strcmp(type, MATMPISBAIJ) == 0) {
        for (int k = 0; k < nnz; k++) {
            if (mtx[k].i >= start && mtx[k].i < end && mtx[k].i <= mtx[k].j) {
                if (mtx[k].j >= start && mtx[k].j < end) {
                    d_nnz[mtx[k].i - start]++;
                } else {
                    o_nnz[mtx[k].i - start]++;
                }
            }
        }
        PetscCall(MatSeqSBAIJSetPreallocation(A, 1, 0, d_nnz));
        PetscCall(MatMPISBAIJSetPreallocation(A, 1, 0, d_nnz, 0, o_nnz));
        free(d_nnz);
        free(o_nnz);
        PetscCall(MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_FALSE));
        for (int k = 0; k < nnz; k++) {
            if (mtx[k].i >= start && mtx[k].i < end && mtx[k].i <= mtx[k].j) {
                PetscCall(MatSetValue(A, mtx[k].i, mtx[k].j, mtx[k].a, INSERT_VALUES));
            }
        }
    } else {
        for (int k = 0; k < nnz; k++) {
            if (mtx[k].i >= start && mtx[k].i < end) {
                if (mtx[k].j >= start && mtx[k].j < end) {
                    d_nnz[mtx[k].i - start]++;
                } else {
                    o_nnz[mtx[k].i - start]++;
                }
            }
        }
        PetscCall(MatSeqAIJSetPreallocation(A, 0, d_nnz));
        PetscCall(MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz));
        PetscCall(MatSeqSELLSetPreallocation(A, 0, d_nnz));
        PetscCall(MatMPISELLSetPreallocation(A, 0, d_nnz, 0, o_nnz));
        free(d_nnz);
        free(o_nnz);
        for (int k = 0; k < nnz; k++) {
            if (mtx[k].i >= start && mtx[k].i < end) {
                PetscCall(MatSetValue(A, mtx[k].i, mtx[k].j, mtx[k].a, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct petsc_mat *create_petsc(int rows, int columns, int nnz, struct mtx *mtx)
{
    struct petsc_mat *petsc = SPMV_ALLOC(struct petsc_mat, 1);
    PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, &petsc->A));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(petsc->A, PETSC_DECIDE, PETSC_DECIDE, rows, columns));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(petsc->A));
    assembly_petsc_mat(petsc->A, rows, columns, nnz, mtx);
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(petsc->A, &petsc->x, &petsc->y));
    return petsc;
}

#endif
