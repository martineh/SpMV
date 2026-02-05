MTX_DIR=~/mtx

# export OMP_PROC_BIND=true
# export OMP_PROC_BIND=spread
# export OMP_PLACES=numa_domains

for i in $MTX_DIR/*
do
    l=`basename $i`
    m=$i/`basename $i`.mtx

    for nt in 1 2 4 8 16
    do
    	export OMP_NUM_THREADS=$nt
        ./spmv_power csr $m >> csr.out
    done

done
