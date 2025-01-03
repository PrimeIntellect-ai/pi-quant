# Kernel Signature:

# extern "C" void kern(
#       const float* const x,
#       uint8_t* const o,
#       const size_t n,
#       const float inv_scale,
#       const int32_t zero_point
# );

# We use AVX512 so 512-bit ZMM0-ZMM31
# Machine encoding is x86-64, 64-bit mode, little-endian, EVEX prefix for ZMM registers.
# Callconv: cdecl System V AMD64 ABI

.globl f32_q8_kernel_vec_avx512
f32_q8_kernel_vec_avx512:
        vmovd   %ecx, %xmm9
        movq    %rsi, %r8
        vmovss  %xmm0, %xmm0, %xmm14
        movq    %rdx, %rsi
        vmovss  %xmm9, %xmm9, %xmm15
        testq   %rdx, %rdx
        je      .L30
        leaq    -1(%rdx), %rax
        cmpq    $14, %rax
        jbe     .L13
        leaq    (%rdi,%rdx,4), %rdx
        cmpq    %rdx, %r8
        jnb     .L15
        leaq    (%r8,%rsi), %rdx
        cmpq    %rdx, %rdi
        jb      .L13
.L15:
        cmpq    $30, %rax
        jbe     .L14
        movl    $255, %r9d
        movq    %rsi, %rcx
        movq    %rdi, %rax
        movq    %r8, %rdx
        vmovd   %r9d, %xmm2
        movl    $65535, %r9d
        shrq    $5, %rcx
        vbroadcastss    .LC1(%rip), %ymm5
        vmovd   %r9d, %xmm1
        movl    $16711935, %r9d
        salq    $7, %rcx
        vbroadcastss    .LC3(%rip), %ymm4
        vmovd   %r9d, %xmm8
        vbroadcastss    %xmm14, %ymm7
        vpbroadcastd    %xmm15, %ymm6
        addq    %rdi, %rcx
        vpxor   %xmm3, %xmm3, %xmm3
        vpbroadcastd    %xmm2, %ymm2
        vpbroadcastd    %xmm1, %ymm1
        vpbroadcastd    %xmm8, %ymm8
.L6:
        vmulps  (%rax), %ymm7, %ymm10
        vmovaps %ymm5, %ymm13
        subq    $-128, %rax
        addq    $32, %rdx
        vmulps  -96(%rax), %ymm7, %ymm12
        vmulps  -64(%rax), %ymm7, %ymm0
        vmulps  -32(%rax), %ymm7, %ymm11
        vpternlogd      $248, %zmm4, %zmm10, %zmm13
        vaddps  %ymm13, %ymm10, %ymm10
        vmovaps %ymm5, %ymm13
        vpternlogd      $248, %zmm4, %zmm12, %zmm13
        vaddps  %ymm13, %ymm12, %ymm12
        vmovaps %ymm5, %ymm13
        vpternlogd      $248, %zmm4, %zmm0, %zmm13
        vroundps        $3, %ymm10, %ymm10
        vcvttps2dq      %ymm10, %ymm10
        vpaddd  %ymm6, %ymm10, %ymm10
        vaddps  %ymm13, %ymm0, %ymm0
        vmovaps %ymm5, %ymm13
        vpmaxsd %ymm3, %ymm10, %ymm10
        vpternlogd      $248, %zmm4, %zmm11, %zmm13
        vroundps        $3, %ymm12, %ymm12
        vcvttps2dq      %ymm12, %ymm12
        vpaddd  %ymm6, %ymm12, %ymm12
        vaddps  %ymm13, %ymm11, %ymm11
        vpmaxsd %ymm3, %ymm12, %ymm12
        vroundps        $3, %ymm0, %ymm0
        vcvttps2dq      %ymm0, %ymm0
        vpminsd %ymm2, %ymm10, %ymm10
        vpaddd  %ymm6, %ymm0, %ymm0
        vpmaxsd %ymm3, %ymm0, %ymm0
        vpminsd %ymm2, %ymm12, %ymm12
        vpand   %ymm10, %ymm1, %ymm10
        vroundps        $3, %ymm11, %ymm11
        vcvttps2dq      %ymm11, %ymm11
        vpaddd  %ymm6, %ymm11, %ymm11
        vpminsd %ymm2, %ymm0, %ymm0
        vpmaxsd %ymm3, %ymm11, %ymm11
        vpand   %ymm12, %ymm1, %ymm12
        vpand   %ymm0, %ymm1, %ymm0
        vpminsd %ymm2, %ymm11, %ymm11
        vpackusdw       %ymm12, %ymm10, %ymm10
        vpand   %ymm11, %ymm1, %ymm11
        vpermq  $216, %ymm10, %ymm10
        vpackusdw       %ymm11, %ymm0, %ymm0
        vpand   %ymm10, %ymm8, %ymm10
        vpermq  $216, %ymm0, %ymm0
        vpand   %ymm0, %ymm8, %ymm0
        vpackuswb       %ymm0, %ymm10, %ymm10
        vpermq  $216, %ymm10, %ymm10
        vmovdqu %ymm10, -32(%rdx)
        cmpq    %rax, %rcx
        jne     .L6
        movq    %rsi, %rdx
        andq    $-32, %rdx
        testb   $31, %sil
        je      .L28
        movq    %rsi, %rcx
        subq    %rdx, %rcx
        leaq    -1(%rcx), %rax
        cmpq    $14, %rax
        jbe     .L32
        vzeroupper
.L5:
        leaq    (%rdi,%rdx,4), %rax
        vbroadcastss    %xmm14, %xmm1
        vbroadcastss    .LC1(%rip), %xmm6
        vpbroadcastd    %xmm15, %xmm4
        vmulps  (%rax), %xmm1, %xmm2
        vpxor   %xmm8, %xmm8, %xmm8
        vmulps  16(%rax), %xmm1, %xmm3
        vmovaps %xmm6, %xmm7
        vmulps  32(%rax), %xmm1, %xmm0
        vmulps  48(%rax), %xmm1, %xmm5
        movl    $255, %eax
        vbroadcastss    .LC3(%rip), %xmm1
        vpternlogd      $248, %zmm1, %zmm2, %zmm7
        vaddps  %xmm7, %xmm2, %xmm2
        vmovaps %xmm6, %xmm7
        vpternlogd      $248, %zmm1, %zmm3, %zmm7
        vaddps  %xmm7, %xmm3, %xmm3
        vmovaps %xmm6, %xmm7
        vpternlogd      $248, %zmm1, %zmm5, %zmm6
        vpternlogd      $248, %zmm1, %zmm0, %zmm7
        vaddps  %xmm6, %xmm5, %xmm5
        vroundps        $3, %xmm2, %xmm2
        vcvttps2dq      %xmm2, %xmm2
        vaddps  %xmm7, %xmm0, %xmm0
        vpaddd  %xmm4, %xmm2, %xmm2
        vmovd   %eax, %xmm7
        movl    $65535, %eax
        vroundps        $3, %xmm3, %xmm3
        vcvttps2dq      %xmm3, %xmm3
        vpaddd  %xmm4, %xmm3, %xmm3
        vpbroadcastd    %xmm7, %xmm7
        vmovd   %eax, %xmm6
        vpmaxsd %xmm8, %xmm2, %xmm2
        vpmaxsd %xmm8, %xmm3, %xmm3
        vroundps        $3, %xmm5, %xmm5
        vroundps        $3, %xmm0, %xmm0
        vcvttps2dq      %xmm5, %xmm5
        vcvttps2dq      %xmm0, %xmm0
        vpaddd  %xmm4, %xmm5, %xmm5
        vpaddd  %xmm4, %xmm0, %xmm0
        vpbroadcastd    %xmm6, %xmm6
        vpminsd %xmm7, %xmm2, %xmm2
        movl    $16711935, %eax
        vpminsd %xmm7, %xmm3, %xmm3
        vpmaxsd %xmm8, %xmm0, %xmm0
        vpand   %xmm2, %xmm6, %xmm2
        vpmaxsd %xmm8, %xmm5, %xmm5
        vpand   %xmm3, %xmm6, %xmm3
        vpminsd %xmm7, %xmm0, %xmm0
        vpminsd %xmm7, %xmm5, %xmm5
        vpackusdw       %xmm3, %xmm2, %xmm2
        vpand   %xmm0, %xmm6, %xmm0
        vmovd   %eax, %xmm3
        vpand   %xmm5, %xmm6, %xmm6
        movq    %rcx, %rax
        vpbroadcastd    %xmm3, %xmm3
        vpackusdw       %xmm6, %xmm0, %xmm0
        andq    $-16, %rax
        vpand   %xmm3, %xmm2, %xmm2
        vpand   %xmm3, %xmm0, %xmm0
        vpackuswb       %xmm0, %xmm2, %xmm2
        vmovdqu %xmm2, (%r8,%rdx)
        addq    %rax, %rdx
        andl    $15, %ecx
        je      .L30
.L8:
        vmulss  (%rdi,%rdx,4), %xmm14, %xmm0
        vpxor   %xmm3, %xmm3, %xmm3
        vmovss  .LC12(%rip), %xmm2
        vmovaps %xmm2, %xmm4
        leaq    0(,%rdx,4), %rcx
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, (%r8,%rdx)
        leaq    1(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  4(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm5
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm5
        vaddss  %xmm5, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 1(%r8,%rdx)
        leaq    2(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  8(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm5
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm5
        vaddss  %xmm5, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 2(%r8,%rdx)
        leaq    3(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  12(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm4
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 3(%r8,%rdx)
        leaq    4(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  16(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm6
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm6
        vaddss  %xmm6, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 4(%r8,%rdx)
        leaq    5(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  20(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm6
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm6
        vaddss  %xmm6, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 5(%r8,%rdx)
        leaq    6(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  24(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm4
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 6(%r8,%rdx)
        leaq    7(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  28(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm5
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm5
        vaddss  %xmm5, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 7(%r8,%rdx)
        leaq    8(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  32(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm5
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm5
        vaddss  %xmm5, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 8(%r8,%rdx)
        leaq    9(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  36(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm4
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 9(%r8,%rdx)
        leaq    10(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  40(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm6
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm6
        vaddss  %xmm6, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 10(%r8,%rdx)
        leaq    11(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  44(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm6
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm6
        vaddss  %xmm6, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 11(%r8,%rdx)
        leaq    12(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  48(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm4
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 12(%r8,%rdx)
        leaq    13(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  52(%rdi,%rcx), %xmm14, %xmm0
        vmovaps %xmm2, %xmm5
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm5
        vaddss  %xmm5, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 13(%r8,%rdx)
        leaq    14(%rdx), %rax
        cmpq    %rsi, %rax
        jnb     .L30
        vmulss  56(%rdi,%rcx), %xmm14, %xmm14
        vpternlogd      $248, %zmm1, %zmm14, %zmm2
        vpxor   %xmm1, %xmm1, %xmm1
        vaddss  %xmm2, %xmm14, %xmm14
        vrndscaless     $3, %xmm14, %xmm14, %xmm14
        vcvttss2sil     %xmm14, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm1, %xmm0, %xmm0
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, 14(%r8,%rdx)
        ret
.L28:
        vzeroupper
.L30:
        ret
.L13:
        vbroadcastss    .LC3(%rip), %xmm1
        xorl    %edx, %edx
        vmovss  .LC12(%rip), %xmm2
.L10:
        vmulss  (%rdi,%rdx,4), %xmm14, %xmm0
        vmovaps %xmm2, %xmm4
        vpxor   %xmm3, %xmm3, %xmm3
        vpternlogd      $248, %zmm1, %zmm0, %zmm4
        vaddss  %xmm4, %xmm0, %xmm0
        vrndscaless     $3, %xmm0, %xmm0, %xmm0
        vcvttss2sil     %xmm0, %eax
        vmovd   %eax, %xmm0
        vpaddd  %xmm9, %xmm0, %xmm0
        vpmaxsd %xmm3, %xmm0, %xmm0
        vmovdqa .LC15(%rip), %xmm3
        vpminsd %xmm3, %xmm0, %xmm0
        vmovd   %xmm0, %eax
        movb    %al, (%r8,%rdx)
        addq    $1, %rdx
        cmpq    %rdx, %rsi
        jne     .L10
        ret
.L14:
        movq    %rsi, %rcx
        xorl    %edx, %edx
        jmp     .L5
.L32:
        vbroadcastss    .LC3(%rip), %xmm1
        vzeroupper
        jmp     .L8
.LC3:
        .long   -2147483648
.LC12:
        .long   1056964607
        .long   0
        .long   0
        .long   0
.LC15:
        .long   255
        .long   0
        .long   0
        .long   0
