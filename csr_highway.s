	.arch armv8-a
	.file	"csr_highway.c"
// GNU C++14 (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) version 7.5.0 (aarch64-linux-gnu)
//	compiled by GNU C version 7.5.0, GMP version 6.1.2, MPFR version 4.0.1, MPC version 1.1.0, isl version isl-0.19-GMP

// GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
// options passed:  -I /nfs/gap/psiwins/highway-xav/highway/
// -imultiarch aarch64-linux-gnu -D_GNU_SOURCE -D_REENTRANT -D NEON
// -D GO_HIGHWAY -D GO_HIGHWAY src/go_highway/csr_highway.c -march=armv8-a
// -mlittle-endian -mabi=lp64 -auxbase-strip csr_highway.s -O3 -Wall
// -fopenmp -fverbose-asm -fstack-protector-strong -Wformat-security
// options enabled:  -fPIC -fPIE -faggressive-loop-optimizations
// -falign-labels -fauto-inc-dec -fbranch-count-reg -fcaller-saves
// -fchkp-check-incomplete-type -fchkp-check-read -fchkp-check-write
// -fchkp-instrument-calls -fchkp-narrow-bounds -fchkp-optimize
// -fchkp-store-bounds -fchkp-use-static-bounds
// -fchkp-use-static-const-bounds -fchkp-use-wrappers -fcode-hoisting
// -fcombine-stack-adjustments -fcommon -fcompare-elim -fcprop-registers
// -fcrossjumping -fcse-follow-jumps -fdefer-pop
// -fdelete-null-pointer-checks -fdevirtualize -fdevirtualize-speculatively
// -fdwarf2-cfi-asm -fearly-inlining -feliminate-unused-debug-types
// -fexceptions -fexpensive-optimizations -fforward-propagate
// -ffp-int-builtin-inexact -ffunction-cse -fgcse -fgcse-after-reload
// -fgcse-lm -fgnu-runtime -fgnu-unique -fguess-branch-probability
// -fhoist-adjacent-loads -fident -fif-conversion -fif-conversion2
// -findirect-inlining -finline -finline-atomics -finline-functions
// -finline-functions-called-once -finline-small-functions -fipa-bit-cp
// -fipa-cp -fipa-cp-clone -fipa-icf -fipa-icf-functions
// -fipa-icf-variables -fipa-profile -fipa-pure-const -fipa-ra
// -fipa-reference -fipa-sra -fipa-vrp -fira-hoist-pressure
// -fira-share-save-slots -fira-share-spill-slots
// -fisolate-erroneous-paths-dereference -fivopts -fkeep-static-consts
// -fleading-underscore -flifetime-dse -flra-remat -flto-odr-type-merging
// -fmath-errno -fmerge-constants -fmerge-debug-strings
// -fmove-loop-invariants -fomit-frame-pointer -foptimize-sibling-calls
// -foptimize-strlen -fpartial-inlining -fpeel-loops -fpeephole -fpeephole2
// -fplt -fpredictive-commoning -fprefetch-loop-arrays -free
// -freg-struct-return -freorder-blocks -freorder-functions
// -frerun-cse-after-loop -fsched-critical-path-heuristic
// -fsched-dep-count-heuristic -fsched-group-heuristic -fsched-interblock
// -fsched-last-insn-heuristic -fsched-pressure -fsched-rank-heuristic
// -fsched-spec -fsched-spec-insn-heuristic -fsched-stalled-insns-dep
// -fschedule-fusion -fschedule-insns -fschedule-insns2 -fsection-anchors
// -fsemantic-interposition -fshow-column -fshrink-wrap
// -fshrink-wrap-separate -fsigned-zeros -fsplit-ivs-in-unroller
// -fsplit-loops -fsplit-paths -fsplit-wide-types -fssa-backprop
// -fssa-phiopt -fstack-protector-strong -fstdarg-opt -fstore-merging
// -fstrict-aliasing -fstrict-overflow -fstrict-volatile-bitfields
// -fsync-libcalls -fthread-jumps -ftoplevel-reorder -ftrapping-math
// -ftree-bit-ccp -ftree-builtin-call-dce -ftree-ccp -ftree-ch
// -ftree-coalesce-vars -ftree-copy-prop -ftree-cselim -ftree-dce
// -ftree-dominator-opts -ftree-dse -ftree-forwprop -ftree-fre
// -ftree-loop-distribute-patterns -ftree-loop-if-convert -ftree-loop-im
// -ftree-loop-ivcanon -ftree-loop-optimize -ftree-loop-vectorize
// -ftree-parallelize-loops= -ftree-partial-pre -ftree-phiprop -ftree-pre
// -ftree-pta -ftree-reassoc -ftree-scev-cprop -ftree-sink
// -ftree-slp-vectorize -ftree-slsr -ftree-sra -ftree-switch-conversion
// -ftree-tail-merge -ftree-ter -ftree-vrp -funit-at-a-time
// -funswitch-loops -fverbose-asm -fzero-initialized-in-bss
// -mfix-cortex-a53-835769 -mfix-cortex-a53-843419 -mglibc -mlittle-endian
// -momit-leaf-frame-pointer -mpc-relative-literal-loads

	.text
	.align	2
	.p2align 3,,7
	.global	mult_csr_highway
	.type	mult_csr_highway, %function
mult_csr_highway:
.LFB9127:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	adrp	x12, :got:__stack_chk_guard	// tmp217,
	add	x29, sp, 0	//,,
	.cfi_def_cfa_register 29
// src/go_highway/csr_highway.c:24: void mult_csr_highway(struct csr *csr, double *x, double *y) {
	ldr	x3, [x12, #:got_lo12:__stack_chk_guard]	// tmp198, tmp217,
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	ldrsw	x11, [x0]	// _53, csr_26(D)->rows
// src/go_highway/csr_highway.c:24: void mult_csr_highway(struct csr *csr, double *x, double *y) {
	ldr	x4, [x3]	// tmp228, __stack_chk_guard
	str	x4, [x29, 56]	// tmp228, D.123031
	mov	x4,0	// tmp228
// src/go_highway/csr_highway.c:32:     double *values = csr->A;
	ldr	x14, [x0, 40]	// values, csr_26(D)->A
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	cmp	x11, 0	// _53,
// src/go_highway/csr_highway.c:31:     int64_t *col_indices = csr->j;
	ldp	x6, x13, [x0, 24]	// row_indices, col_indices, csr_26(D)->i
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	ble	.L1	//,
// src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	movi	d4, #0	// _116
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	mov	x8, 0	// i,
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6493:   return __builtin_aarch64_combinedi (__a[0], __b[0]);
	mov	x15, 0	// tmp223,
	b	.L11	//
	.p2align 2
.L20:
// src/go_highway/csr_highway.c:51:             y[i] = 0.0;  
	str	xzr, [x2, x8, lsl 3]	//, MEM[base: y_42(D), index: _106, step: 8, offset: 0B]
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	add	x8, x8, 1	// i, i,
	add	x6, x6, 8	// ivtmp.199, ivtmp.199,
	cmp	x8, x11	// i, _53
	beq	.L1	//,
.L11:
// src/go_highway/csr_highway.c:48:         int64_t elements = end_index - start_index;
	ldp	x3, x0, [x6]	// start_index, MEM[base: _1, offset: 8B], MEM[base: _1, offset: 0B]
	sub	x0, x0, x3	// elements, MEM[base: _1, offset: 8B], start_index
// src/go_highway/csr_highway.c:50:         if (elements == 0) {
	cmp	x0, 0	// elements,
	beq	.L20	//,
// src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	ble	.L13	//,
	sub	x4, x0, #1	// tmp202, ivtmp.189,
	lsl	x3, x3, 3	// _109, start_index,
	sub	x7, x0, #2	// tmp201, ivtmp.189,
	and	x4, x4, -2	// tmp203, tmp202,
	add	x5, x13, x3	// ivtmp.190, col_indices, _109
	sub	x7, x7, x4	// _51, tmp201, tmp203
// src/go_highway/csr_highway.c:55:         auto prod_v = hn::Zero(d);
	movi	v1.2d, 0	// prod_v$raw
	add	x3, x14, x3	// ivtmp.191, values, _109
	add	x10, x29, 16	// tmp218,,
	add	x9, x29, 32	// tmp219,,
	b	.L10	//
	.p2align 2
.L22:
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:17168:   return __builtin_aarch64_ld1v2df ((const __builtin_aarch64_simd_df *) a);
	ldr	q2, [x3]	// SR.169,* ivtmp.191
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:17220:   return __builtin_aarch64_ld1v2di ((const __builtin_aarch64_simd_di *) a);
	ldr	q0, [x5]	// SR.171,* ivtmp.190
.L8:
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:26929:   __builtin_aarch64_st1v2di ((__builtin_aarch64_simd_di *) a, b);
	str	q0, [x10]	// SR.171,
// /nfs/gap/psiwins/highway-xav/highway/hwy/ops/generic_ops-inl.h:1536:     lanes[i] = base[index_lanes[i]];
	fmov	x16, d0	// tmp230, SR.171
	ldr	x4, [x29, 24]	// index_lanes, index_lanes
	sub	x0, x0, #2	// ivtmp.189, ivtmp.189,
	add	x5, x5, 16	// ivtmp.190, ivtmp.190,
	add	x3, x3, 16	// ivtmp.191, ivtmp.191,
// src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	cmp	x0, x7	// ivtmp.189, _51
// /nfs/gap/psiwins/highway-xav/highway/hwy/ops/generic_ops-inl.h:1536:     lanes[i] = base[index_lanes[i]];
	ldr	d3, [x1, x16, lsl 3]	// *_79, *_79
	ldr	d0, [x1, x4, lsl 3]	// *_88, *_88
	stp	d3, d0, [x29, 32]	// *_79, *_88, lanes
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:17168:   return __builtin_aarch64_ld1v2df ((const __builtin_aarch64_simd_df *) a);
	ldr	q0, [x9]	// _57,
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:16724:   return __builtin_aarch64_fmav2df (__b, __c, __a);
	fmla	v1.2d, v2.2d, v0.2d	// prod_v$raw, SR.169, _57
// src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	beq	.L21	//,
.L10:
// /nfs/gap/psiwins/highway-xav/highway/hwy/ops/generic_ops-inl.h:1116:   if (max_lanes_to_load >= 2) {
	cmp	x0, 1	// ivtmp.189,
	bgt	.L22	//,
// /nfs/gap/psiwins/highway-xav/highway/hwy/ops/generic_ops-inl.h:1121:                : Zero(d);
	movi	v2.2d, 0	// SR.169
	movi	v0.4s, 0	// SR.171
	bne	.L8	//,
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6545:   return __builtin_aarch64_combinedf (__a[0], __b[0]);
	ldr	d2, [x3]	// MEM[base: _14, offset: 0B], MEM[base: _14, offset: 0B]
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6493:   return __builtin_aarch64_combinedi (__a[0], __b[0]);
	ldr	x4, [x5]	// MEM[base: _16, offset: 0B], MEM[base: _16, offset: 0B]
	fmov	d0, x4	// SR.171, MEM[base: _16, offset: 0B]
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6545:   return __builtin_aarch64_combinedf (__a[0], __b[0]);
	dup	d2, v2.d[0]	// SR.169, MEM[base: _14, offset: 0B]
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6493:   return __builtin_aarch64_combinedi (__a[0], __b[0]);
	ins	v0.d[1], x15	// SR.171, tmp223
// /usr/lib/gcc/aarch64-linux-gnu/7/include/arm_neon.h:6545:   return __builtin_aarch64_combinedf (__a[0], __b[0]);
	ins	v2.d[1], v4.d[0]	// SR.169, _116
	b	.L8	//
	.p2align 2
.L21:
	faddp	d1, v1.2d	// _116, prod_v$raw
// src/go_highway/csr_highway.c:79:         y[i] = hn::ReduceSum(d, prod_v) ;
	str	d1, [x2, x8, lsl 3]	// _116, MEM[base: y_42(D), index: _107, step: 8, offset: 0B]
.L24:
// src/go_highway/csr_highway.c:45:     for (int64_t i = 0; i < nrows; i++) {
	add	x8, x8, 1	// i, i,
	add	x6, x6, 8	// ivtmp.199, ivtmp.199,
	cmp	x8, x11	// i, _53
	bne	.L11	//,
.L1:
// src/go_highway/csr_highway.c:81:   }
	ldr	x12, [x12, #:got_lo12:__stack_chk_guard]	// tmp214, tmp217,
	ldr	x1, [x29, 56]	// tmp229, D.123031
	ldr	x0, [x12]	// tmp216, __stack_chk_guard
	eor	x0, x1, x0	// tmp216, tmp229
	cbnz	x0, .L23	// tmp216,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.p2align 2
.L13:
	.cfi_restore_state
// src/go_highway/csr_highway.c:59:         for (int64_t j = 0; j < elements; ) {
	movi	d1, #0	// _116
// src/go_highway/csr_highway.c:79:         y[i] = hn::ReduceSum(d, prod_v) ;
	str	d1, [x2, x8, lsl 3]	// _116, MEM[base: y_42(D), index: _107, step: 8, offset: 0B]
	b	.L24	//
.L23:
// src/go_highway/csr_highway.c:81:   }
	bl	__stack_chk_fail	//
	.cfi_endproc
.LFE9127:
	.size	mult_csr_highway, .-mult_csr_highway
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 3,,7
	.type	_GLOBAL__sub_I_mult_csr_highway, %function
_GLOBAL__sub_I_mult_csr_highway:
.LFB10726:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0	//,,
	.cfi_def_cfa_register 29
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// /usr/include/c++/7/iostream:74:   static ios_base::Init __ioinit;
	adrp	x19, .LANCHOR0	// tmp74,
	add	x19, x19, :lo12:.LANCHOR0	// tmp73, tmp74,
	mov	x0, x19	//, tmp73
	bl	_ZNSt8ios_base4InitC1Ev	//
	adrp	x0, :got:_ZNSt8ios_base4InitD1Ev	// tmp80,
	mov	x1, x19	//, tmp73
// src/go_highway/csr_highway.c:83: } // extern "C"
	ldr	x19, [sp, 16]	//,
// /usr/include/c++/7/iostream:74:   static ios_base::Init __ioinit;
	adrp	x2, __dso_handle	// tmp76,
	ldr	x0, [x0, #:got_lo12:_ZNSt8ios_base4InitD1Ev]	//, tmp80,
	add	x2, x2, :lo12:__dso_handle	//, tmp76,
// src/go_highway/csr_highway.c:83: } // extern "C"
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa 31, 0
// /usr/include/c++/7/iostream:74:   static ios_base::Init __ioinit;
	b	__cxa_atexit	//
	.cfi_endproc
.LFE10726:
	.size	_GLOBAL__sub_I_mult_csr_highway, .-_GLOBAL__sub_I_mult_csr_highway
	.section	.init_array,"aw"
	.align	3
	.xword	_GLOBAL__sub_I_mult_csr_highway
	.bss
	.align	3
	.set	.LANCHOR0,. + 0
	.type	_ZStL8__ioinit, %object
	.size	_ZStL8__ioinit, 1
_ZStL8__ioinit:
	.zero	1
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) 7.5.0"
	.section	.note.GNU-stack,"",@progbits
