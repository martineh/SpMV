	.file	"csr_highway.c"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zfh1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
# GNU C++17 (Bianbu 13.2.0-23ubuntu4bb3) version 13.2.0 (riscv64-linux-gnu)
#	compiled by GNU C version 13.2.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=spacemit-x60 -mabi=lp64d -misa-spec=20191213 -march=rv64imafdcv_zicsr_zifencei_zfh_zfhmin_zve32f_zve32x_zve64d_zve64f_zve64x_zvl128b_zvl32b_zvl64b -O3 -fopenmp -fstack-protector-strong
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.align	1
	.globl	mult_csr_highway
	.type	mult_csr_highway, @function
mult_csr_highway:
.LFB9893:
	.cfi_startproc
# src/go_highway/csr_highway.c:27:     int nrows = csr->rows;
	lw	t4,0(a0)		# nrows, csr_25(D)->rows
# src/go_highway/csr_highway.c:24: void mult_csr_highway(struct csr *csr, double *x, double *y) {
	addi	sp,sp,-32	#,,
	.cfi_def_cfa_offset 32
	lui	t0,%hi(__stack_chk_guard)	# tmp194,
	sd	ra,24(sp)	#,
	.cfi_offset 1, -8
# src/go_highway/csr_highway.c:24: void mult_csr_highway(struct csr *csr, double *x, double *y) {
	ld	a5, %lo(__stack_chk_guard)(t0)	# tmp205, __stack_chk_guard
	sd	a5, 8(sp)	# tmp205, D.213007
	li	a5, 0	# tmp205
# src/go_highway/csr_highway.c:29:     int64_t* row_indices =csr->i;
	ld	t3,24(a0)		# row_indices, csr_25(D)->i
# src/go_highway/csr_highway.c:31:     int64_t *col_indices = csr->j;
	ld	t5,32(a0)		# col_indices, csr_25(D)->j
# src/go_highway/csr_highway.c:32:     double *values = csr->A;
	ld	t6,40(a0)		# values, csr_25(D)->A
# src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	ble	t4,zero,.L1	#, nrows,,
	vsetvli	a5,zero,e64,m2,ta,mu	#, _41,,,,
	slli	t4,t4,3	#, tmp189, nrows
	slli	t1,a5,3	#, _101, _41
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:732: HWY_RVV_FOREACH_UI(HWY_RVV_SET, Set, mv_v_x, _ALL_VIRT)
	vsetvli	a5,zero,e64,m2,ta,ma	#,,,,,
	add	t4,t3,t4	# tmp189, _21, ivtmp.96
	vmv.v.i	v28,0	# _30,
	vsetvli	a4,zero,e64,m1,ta,ma	#,,,,,
	vmv.v.i	v8,0	# _81,
	j	.L7		#
.L3:
# src/go_highway/csr_highway.c:51:             y[i] = 0.0;  
	fsd	fa5,0(a2)	# _86, MEM[(double *)_56]
# src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	addi	t3,t3,8	#, ivtmp.96, ivtmp.96
	addi	a2,a2,8	#, ivtmp.97, ivtmp.97
	beq	t4,t3,.L1	#, _21, ivtmp.96,
.L7:
# src/go_highway/csr_highway.c:46:         const int64_t start_index = row_indices[i];
	ld	a3,0(t3)		# start_index, MEM[(int64_t *)_59]
	fmv.d.x	fa5,zero	# _86,
# src/go_highway/csr_highway.c:48:         int64_t elements = end_index - start_index;
	ld	a4,8(t3)		# MEM[(int64_t *)_59 + 8B], MEM[(int64_t *)_59 + 8B]
	sub	a7,a4,a3	# elements, MEM[(int64_t *)_59 + 8B], start_index
# src/go_highway/csr_highway.c:50:         if (elements == 0) {
	beq	a4,a3,.L3	#, MEM[(int64_t *)_59 + 8B], start_index,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:936: HWY_RVV_FOREACH_F(HWY_RVV_CAST_IF, _, reinterpret, _ALL)
	vmv2r.v	v26,v28	# _30, prod_v
# src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	ble	a7,zero,.L16	#, elements,,
	slli	a3,a3,3	#, _95, start_index
# src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	li	a0,0		# j,
	add	a6,t6,a3	# _95, ivtmp.88, values
	add	a3,t5,a3	# _95, ivtmp.89, col_indices
.L6:
# src/go_highway/csr_highway.c:60:             int64_t restantes = elements - j;
	sub	a4,a7,a0	# _77, elements, j
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:657: HWY_RVV_FOREACH(HWY_RVV_LANES, Lanes, setvlmax_e, _ALL)
	bleu	a4,a5,.L5	#, _77, _41,
	mv	a4,a5	# _77, _41
.L5:
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:2061: HWY_RVV_FOREACH(HWY_RVV_LOADN, LoadN, le, _ALL_VIRT)
	vsetvli	zero,a4,e64,m2,tu,ma	# _77,,,,
	add	a0,a0,a5	# _41, j, j
	vmv2r.v	v30,v28	# _30, _97
	vmv2r.v	v24,v28	# _30, _64
	vle64.v	v30,0(a6)	# _97,* ivtmp.88,
	vle64.v	v24,0(a3)	# _64,* ivtmp.89,
# src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	add	a6,a6,t1	# _101, ivtmp.88, ivtmp.88
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:1255: HWY_RVV_FOREACH_UI(HWY_RVV_SHIFT, ShiftLeft, sll, _ALL)
	vsetvli	a5,zero,e64,m2,ta,ma	#,,,,,
# src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	add	a3,a3,t1	# _101, ivtmp.89, ivtmp.89
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:1255: HWY_RVV_FOREACH_UI(HWY_RVV_SHIFT, ShiftLeft, sll, _ALL)
	vsll.vi	v24,v24,3	#, _47, _64,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:2247: HWY_RVV_FOREACH(HWY_RVV_GATHER, GatherOffset, lux, _ALL_VIRT)
	vluxei64.v	v24,(a1),v24	# _51, x, _47,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:1571: HWY_RVV_FOREACH_F(HWY_RVV_FMA, MulAdd, fmacc, _ALL)
	vfmacc.vv	v26,v30,v24	# prod_v, _97, _51,
# src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	bgt	a7,a0,.L6	#, elements, j,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:4800: HWY_RVV_FOREACH_F(HWY_RVV_REDUCE, RedSum, fredusum, _ALL_VIRT)
	vfredusum.vs	v26,v26,v8	# _85, prod_v, _81,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:3540: HWY_RVV_FOREACH_F(HWY_RVV_GET_LANE, GetLane, fmv_f, _ALL)
	vfmv.f.s	fa5,v26	# _86, _85
.L18:
# src/go_highway/csr_highway.c:51:             y[i] = 0.0;  
	fsd	fa5,0(a2)	# _86, MEM[(double *)_56]
# src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	addi	t3,t3,8	#, ivtmp.96, ivtmp.96
	addi	a2,a2,8	#, ivtmp.97, ivtmp.97
	bne	t4,t3,.L7	#, _21, ivtmp.96,
.L1:
# src/go_highway/csr_highway.c:81:   }
	ld	a4, 8(sp)	# tmp206, D.213007
	ld	a5, %lo(__stack_chk_guard)(t0)	# tmp193, __stack_chk_guard
	xor	a5, a4, a5	# tmp193, tmp206
	li	a4, 0	# tmp206
	bne	a5,zero,.L17	#, tmp193,,
	ld	ra,24(sp)		#,
	.cfi_remember_state
	.cfi_restore 1
	addi	sp,sp,32	#,,
	.cfi_def_cfa_offset 0
	jr	ra		#
.L16:
	.cfi_restore_state
	vsetvli	a5,zero,e64,m2,ta,ma	#,,,,,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:4800: HWY_RVV_FOREACH_F(HWY_RVV_REDUCE, RedSum, fredusum, _ALL_VIRT)
	vfredusum.vs	v26,v26,v8	# _85, prod_v, _81,
# /nfs/gap/psiwins/highway/hwy/ops/rvv-inl.h:3540: HWY_RVV_FOREACH_F(HWY_RVV_GET_LANE, GetLane, fmv_f, _ALL)
	vfmv.f.s	fa5,v26	# _86, _85
	j	.L18		#
.L17:
# src/go_highway/csr_highway.c:81:   }
	call	__stack_chk_fail		#
	.cfi_endproc
.LFE9893:
	.size	mult_csr_highway, .-mult_csr_highway
	.ident	"GCC: (Bianbu 13.2.0-23ubuntu4bb3) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
