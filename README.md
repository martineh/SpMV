SpMV
====

`spmv_power.c` es el driver principal. El primer parámetro es la implementación del producto (corresponde con el nombre del módulo más abajo) y el segundo el fichero MTX. Cada campo de la línea de salida es:
1. Implementación SpMV
2. Nombre de la matriz
3. Hilos de OpenMP
4. Procesos de MPI
5. Filas
6. Columnas
7. Valores no nulos
8. FLOP/s del SpMV
9. Tiempo medio del SpMV
10. Iteraciones
11. Error comparando con COO
12. Memoria utilizada

Este programa calcula el método de la potencia, es decir multiplica un vector normalizado por la matriz, el resultado se normaliza y repite la multiplicación. La idea es que sea más realista que simplemente repetir la SpMV una y otra vez que como la SpMV depende mucho de la caché hace que la SpMV funcione demasiado bien.

`ssstats.txt`: Lista de matrices (para bajar con wget)
`run.sh`: script para ejecutar

Implementaciones del SPMV
-------------------------

`csr.c`: CSR básico. He hecho una versión con intrínsecas de AVX2 que funciona peor que el código hecho por GCC :-(

`csr_merge.c`: código de Enrique para el CSR merge

`csr_bal.c`: versión alternativa del CSR merge que es más fácil de vectorizar. De hecho la vectorización es la misma que el CSR normal.

`csr_epi.c`: CSR vectorizada para el EPI del BSC

`sell.c`, `radix_sort.c`, `sellcs_analyzer.c`, `sellcs_format.c`, `sellcs_mv_autovector.c`, `sellcs_mv.c`, `sellcs_mv_kernels_epi.c`, `sellcs_utils.c`: código del BSC para el formato SELL (versión vectorizada solamente para el EPI)

`csr_numa.c`: CSR copiando la matriz en trozos para cada thread

`coo.c`: COO

`csr_mkl.c`: CSR de la MKL

`petsc.c`: SpMV del PETSc

`csri.c`, `ell0.c`, `pcsr.c`, `mtx_to_bin.c`: experimentos NO MIRAR ;-)

Otros ficheros
--------------

`spmv.h`: cabecera con las definiciones de todos los módulos

`mtx.c`: código para cargar matrices
