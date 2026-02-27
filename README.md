
# SpMV Driver

SpMV Driver is an application designed for the evaluation of the Sparse Matrix-Vector multiplication (SpMV) operation.

This software enables users to perform systematic benchmarking and analysis of different sparse matrix storage formats and implementation strategies within the target system. It provides a structured environment to compare performance across multiple approaches for executing the sparse matrix-vector product.

The tool is intended to support experimentation, optimization, and performance evaluation of SpMV implementations, helping users identify the most efficient configuration for their specific hardware and workload.

## Requisites

Installation of the Google Highway library is mandatory if you intend to perform an evaluation based on this library.

## Supported Hardware

Any processor with this architectures:

- ARM Neon
- ARM SVE
- ARM SVE2
- AVX2
- RISCV-V RVV 1.0 (256 bits vector length)

## How to use

1. Configure Makefile.inc with the compilation options (autovectorization, platform and Google Highway integration).
2. Compile the library with "make"
3. Configure "spmv.run" with the matrix list to test and the SpMV algorithm to be tested. 
4. Execute "./spmv.run"
5. The results will appear on the screen, and an output file will be generated.
 
