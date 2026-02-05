#include <stdio.h>
#include <unistd.h>
#include <sys/times.h>

#include "spmv.h"

int main(int argc, char *argv[])
{
    int rows, columns, nnz;
    struct mtx *coo;

    if (argc != 3) {
        printf("Usage: mtx_2_bin <mtx file> <output file>\n");
        return 1;
    }

    printf("Reading %s...\n", argv[1]);
    load_mtx(argv[1], &rows, &columns, &nnz, &coo);
    struct tms t1;
    times(&t1);
    float tick =  sysconf(_SC_CLK_TCK);
    printf("Time: %f %f\n", t1.tms_utime / tick, t1.tms_stime / tick);

    printf("Sorting ..\n");
    sort_mtx(nnz, coo, by_row_mtx);
    struct tms t2;
    times(&t2);
    printf("Time: %f %f\n", (t2.tms_utime - t1.tms_utime) / tick,
            (t2.tms_stime - t1.tms_stime) / tick);

    printf("Writing %s...\n", argv[2]);
    save_bin(argv[2], rows, columns, nnz, coo);
    struct tms t3;
    times(&t3);
    printf("Time: %f %f\n", (t3.tms_utime - t2.tms_utime) / tick,
            (t3.tms_stime - t2.tms_stime) / tick);

    return 0;
}

