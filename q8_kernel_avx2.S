# Kernel Signature:

# extern "C" void kern(
#       const float* const x,
#       uint8_t* const o,
#       const size_t n,
#       const float inv_scale,
#       const int32_t zero_point
# );

# We use AVX2 so 256-bit YMM0-YMM15 (can be extended to 512-bit ZMM0-ZMM31 or YMMO-YMM31 for AVX256).
# Machine encoding is x86-64, 64-bit mode, little-endian, VEX prefix (VEX.256.66.0F.WIG.29.C8) for YMM registers.
# Callconv: cdecl System V AMD64 ABI

.globl f32_q8_kernel_vec_avx2
f32_q8_kernel_vec_avx2:
        vmovd   %ecx, %xmm10
        movq    %rsi, %r8
        vmovaps %xmm0, %xmm15
        movq    %rdx, %rsi
        vmovdqa %xmm10, %xmm11
        testq   %rdx, %rdx
        je      .DONE
        leaq    -1(%rdx), %rax
        cmpq    $14, %rax
        jbe     .KPLOAD3
        leaq    (%rdi,%rdx,4), %rdx
        cmpq    %rdx, %r8
        jnb     .MAIN
        leaq    (%r8,%rsi), %rdx
        cmpq    %rdx, %rdi
        jb      .KPLOAD3
.MAIN:
        cmpq    $30, %rax
        jbe     .STITCH_SKIP
        movl    $255, %r9d
        movq    %rsi, %rcx
        movq    %rdi, %rax
        movq    %r8, %rdx
        vmovd   %r9d, %xmm3
        movl    $65535, %r9d
        shrq    $5, %rcx
        vbroadcastss    .LC1(%rip), %ymm6
        vmovd   %r9d, %xmm2
        movl    $16711935, %r9d
        salq    $7, %rcx
        vbroadcastss    .LC3(%rip), %ymm5
        vmovd   %r9d, %xmm9
        vbroadcastss    %xmm15, %ymm8
        vpxor   %xmm4, %xmm4, %xmm4
        addq    %rdi, %rcx
        vpbroadcastd    %xmm11, %ymm7
        vpbroadcastd    %xmm3, %ymm3
        vpbroadcastd    %xmm2, %ymm2
        vpbroadcastd    %xmm9, %ymm9
.VECTOR_LOOP:
        vmulps  (%rax), %ymm8, %ymm0
        subq    $-128, %rax
        addq    $32, %rdx
        vmulps  -96(%rax), %ymm8, %ymm13
        vmulps  -64(%rax), %ymm8, %ymm1
        vmulps  -32(%rax), %ymm8, %ymm12
        vandps  %ymm0, %ymm5, %ymm14
        vorps   %ymm14, %ymm6, %ymm14
        vaddps  %ymm14, %ymm0, %ymm0
        vandps  %ymm13, %ymm5, %ymm14
        vorps   %ymm14, %ymm6, %ymm14
        vaddps  %ymm14, %ymm13, %ymm13
        vandps  %ymm1, %ymm5, %ymm14
        vorps   %ymm14, %ymm6, %ymm14
        vroundps        $3, %ymm0, %ymm0
        vcvttps2dq      %ymm0, %ymm0
        vpaddd  %ymm7, %ymm0, %ymm0
        vaddps  %ymm14, %ymm1, %ymm1
        vandps  %ymm12, %ymm5, %ymm14
        vpmaxsd %ymm4, %ymm0, %ymm0
        vorps   %ymm14, %ymm6, %ymm14
        vroundps        $3, %ymm13, %ymm13
        vcvttps2dq      %ymm13, %ymm13
        vpaddd  %ymm7, %ymm13, %ymm13
        vaddps  %ymm14, %ymm12, %ymm12
        vpmaxsd %ymm4, %ymm13, %ymm13
        vroundps        $3, %ymm1, %ymm1
        vcvttps2dq      %ymm1, %ymm1
        vpaddd  %ymm7, %ymm1, %ymm1
        vpminsd %ymm3, %ymm0, %ymm0
        vpmaxsd %ymm4, %ymm1, %ymm1
        vpminsd %ymm3, %ymm13, %ymm13
        vpand   %ymm0, %ymm2, %ymm0
        vroundps        $3, %ymm12, %ymm12
        vcvttps2dq      %ymm12, %ymm12
        vpaddd  %ymm7, %ymm12, %ymm12
        vpminsd %ymm3, %ymm1, %ymm1
        vpmaxsd %ymm4, %ymm12, %ymm12
        vpand   %ymm13, %ymm2, %ymm13
        vpand   %ymm1, %ymm2, %ymm1
        vpminsd %ymm3, %ymm12, %ymm12
        vpackusdw       %ymm13, %ymm0, %ymm0
        vpand   %ymm12, %ymm2, %ymm12
        vpermq  $216, %ymm0, %ymm0
        vpackusdw       %ymm12, %ymm1, %ymm1
        vpand   %ymm0, %ymm9, %ymm0
        vpermq  $216, %ymm1, %ymm1
        vpand   %ymm1, %ymm9, %ymm1
        vpackuswb       %ymm1, %ymm0, %ymm0
        vpermq  $216, %ymm0, %ymm0
        vmovdqu %ymm0, -32(%rdx)
        cmpq    %rax, %rcx
        jne     .VECTOR_LOOP
        movq    %rsi, %rdx
        andq    $-32, %rdx
        testb   $31, %sil
        je      .DONE_V
        movq    %rsi, %rcx
        subq    %rdx, %rcx
        leaq    -1(%rcx), %rax
        cmpq    $14, %rax
        jbe     .TAIL_SKIP
        vzeroupper
.PARTIAL_LOOP:
        leaq    (%rdi,%rdx,4), %rax
        vbroadcastss    %xmm15, %xmm1
        vbroadcastss    .LC3(%rip), %xmm5
        vbroadcastss    .LC1(%rip), %xmm4
        vmulps  16(%rax), %xmm1, %xmm2
        vpbroadcastd    %xmm11, %xmm11
        vmulps  32(%rax), %xmm1, %xmm0
        vmulps  48(%rax), %xmm1, %xmm3
        vmulps  (%rax), %xmm1, %xmm1
        movl    $255, %eax
        vandps  %xmm1, %xmm5, %xmm6
        vorps   %xmm6, %xmm4, %xmm6
        vaddps  %xmm6, %xmm1, %xmm1
        vandps  %xmm2, %xmm5, %xmm6
        vorps   %xmm6, %xmm4, %xmm6
        vaddps  %xmm6, %xmm2, %xmm2
        vandps  %xmm0, %xmm5, %xmm6
        vandps  %xmm3, %xmm5, %xmm5
        vorps   %xmm6, %xmm4, %xmm6
        vorps   %xmm5, %xmm4, %xmm4
        vroundps        $3, %xmm1, %xmm1
        vpxor   %xmm5, %xmm5, %xmm5
        vaddps  %xmm4, %xmm3, %xmm3
        vcvttps2dq      %xmm1, %xmm1
        vmovd   %eax, %xmm4
        vpaddd  %xmm11, %xmm1, %xmm1
        movl    $65535, %eax
        vroundps        $3, %xmm2, %xmm2
        vcvttps2dq      %xmm2, %xmm2
        vpaddd  %xmm11, %xmm2, %xmm2
        vpbroadcastd    %xmm4, %xmm4
        vaddps  %xmm6, %xmm0, %xmm0
        vpmaxsd %xmm5, %xmm1, %xmm1
        vmovd   %eax, %xmm6
        vpmaxsd %xmm5, %xmm2, %xmm2
        vpminsd %xmm4, %xmm1, %xmm1
        vroundps        $3, %xmm3, %xmm3
        vpbroadcastd    %xmm6, %xmm6
        vpminsd %xmm4, %xmm2, %xmm2
        movl    $16711935, %eax
        vpand   %xmm2, %xmm6, %xmm2
        vpand   %xmm1, %xmm6, %xmm1
        vroundps        $3, %xmm0, %xmm0
        vcvttps2dq      %xmm0, %xmm0
        vpackusdw       %xmm2, %xmm1, %xmm1
        vpaddd  %xmm11, %xmm0, %xmm0
        vcvttps2dq      %xmm3, %xmm2
        vpaddd  %xmm11, %xmm2, %xmm2
        vpmaxsd %xmm5, %xmm0, %xmm0
        vpmaxsd %xmm5, %xmm2, %xmm2
        vpminsd %xmm4, %xmm2, %xmm2
        vpminsd %xmm4, %xmm0, %xmm0
        vpand   %xmm0, %xmm6, %xmm0
        vpand   %xmm2, %xmm6, %xmm6
        vmovd   %eax, %xmm2
        movq    %rcx, %rax
        vpbroadcastd    %xmm2, %xmm2
        vpackusdw       %xmm6, %xmm0, %xmm0
        andq    $-16, %rax
        vpand   %xmm2, %xmm1, %xmm1
        vpand   %xmm2, %xmm0, %xmm0
        vpackuswb       %xmm0, %xmm1, %xmm1
        vmovdqu %xmm1, (%r8,%rdx)
        addq    %rax, %rdx
        andl    $15, %ecx
        je      .DONE
.TAIL:
        vmulss  (%rdi,%rdx,4), %xmm15, %xmm0
        vmovss  .K1(%rip), %xmm1
        vmovss  .K2(%rip), %xmm2
        leaq    0(,%rdx,4), %rcx
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, (%r8,%rdx)
        leaq    1(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  4(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 1(%r8,%rdx)
        leaq    2(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  8(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 2(%r8,%rdx)
        leaq    3(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  12(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 3(%r8,%rdx)
        leaq    4(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  16(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 4(%r8,%rdx)
        leaq    5(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  20(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 5(%r8,%rdx)
        leaq    6(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  24(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 6(%r8,%rdx)
        leaq    7(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  28(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 7(%r8,%rdx)
        leaq    8(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  32(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 8(%r8,%rdx)
        leaq    9(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  36(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 9(%r8,%rdx)
        leaq    10(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  40(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 10(%r8,%rdx)
        leaq    11(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  44(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 11(%r8,%rdx)
        leaq    12(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  48(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 12(%r8,%rdx)
        leaq    13(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  52(%rdi,%rcx), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 13(%r8,%rdx)
        leaq    14(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .DONE
        vmulss  56(%rdi,%rcx), %xmm15, %xmm15
        vandps  %xmm15, %xmm1, %xmm0
        vpxor   %xmm1, %xmm1, %xmm1
        vorps   %xmm0, %xmm2, %xmm0
        vaddss  %xmm0, %xmm15, %xmm15
        vroundss        $3, %xmm15, %xmm15, %xmm15
        vcvttss2sil     %xmm15, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm1, %xmm0, %xmm0
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 14(%r8,%rdx)
        ret
.DONE_V:
        vzeroupper
.DONE:
        ret
.KPLOAD3:
        xorl    %edx, %edx
        vmovss  .K1(%rip), %xmm1
        vmovss  .K2(%rip), %xmm2
.PARTIAL_STITCH:
        vmulss  (%rdi,%rdx,4), %xmm15, %xmm0
        vandps  %xmm0, %xmm1, %xmm3
        vorps   %xmm3, %xmm2, %xmm3
        vaddss  %xmm3, %xmm0, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vroundss        $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm10, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .K3(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, (%r8,%rdx)
        incq    %rdx
        cmpq    %rdx, %rsi
        jne     .PARTIAL_STITCH
        ret
.STITCH_SKIP:
        movq    %rsi, %rcx
        xorl    %edx, %edx
        jmp     .PARTIAL_LOOP
.TAIL_SKIP:
        vzeroupper
        jmp     .TAIL
.K1:
        .long   -2147483648
        .long   0
        .long   0
        .long   0
.K2:
        .long   1056964607
        .long   0
        .long   0
        .long   0
.K3:
        .long   255
        .long   0
        .long   0
        .long   0
