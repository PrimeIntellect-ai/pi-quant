# Kernel Signature:

# extern "C" void kern(
#       const float* const x,
#       uint8_t* const o,
#       const size_t n,
#       const float inv_scale,
#       const int32_t zero_point
# );

# We use SSE4.1 so 128-bit XMM0-XMM15
# Machine encoding is x86-64, 64-bit mode, little-endian, REX+Legacy SSE prefix.
# Callconv: cdecl System V AMD64 ABI

.globl f32_q8_kernel_vec_sse41
f32_q8_kernel_vec_sse41:
        movd    %ecx, %xmm10
        movq    %rsi, %r8
        movaps  %xmm0, %xmm15
        movq    %rdx, %rsi
        movdqa  %xmm10, %xmm11
        testq   %rdx, %rdx
        je      .L1
        leaq    -1(%rdx), %rax
        cmpq    $6, %rax
        jbe     .L13
        leaq    (%rdi,%rdx,4), %rdx
        cmpq    %rdx, %r8
        jnb     .L15
        leaq    (%r8,%rsi), %rdx
        cmpq    %rdx, %rdi
        jb      .L13
.L15:
        cmpq    $14, %rax
        jbe     .L14
        movq    %rsi, %rcx
        movl    $255, %r9d
        movaps  %xmm15, %xmm4
        movq    %rdi, %rax
        shrq    $4, %rcx
        movl    $65535, %r10d
        movq    %r8, %rdx
        movss   .LC1(%rip), %xmm2
        movss   .LC3(%rip), %xmm3
        salq    $6, %rcx
        movl    $16711935, %r11d
        movd    %r9d, %xmm6
        movd    %r10d, %xmm5
        movd    %r11d, %xmm9
        addq    %rdi, %rcx
        shufps  $0, %xmm4, %xmm4
        pxor    %xmm7, %xmm7
        pshufd  $0, %xmm11, %xmm8
        shufps  $0, %xmm2, %xmm2
        shufps  $0, %xmm3, %xmm3
        pshufd  $0, %xmm6, %xmm6
        pshufd  $0, %xmm5, %xmm5
        pshufd  $0, %xmm9, %xmm9
.L6:
        movups  (%rax), %xmm0
        movups  16(%rax), %xmm13
        movaps  %xmm3, %xmm14
        addq    $64, %rax
        movups  -32(%rax), %xmm1
        movups  -16(%rax), %xmm12
        addq    $16, %rdx
        mulps   %xmm4, %xmm0
        mulps   %xmm4, %xmm13
        mulps   %xmm4, %xmm1
        mulps   %xmm4, %xmm12
        andps   %xmm0, %xmm14
        orps    %xmm2, %xmm14
        addps   %xmm14, %xmm0
        movaps  %xmm3, %xmm14
        andps   %xmm13, %xmm14
        orps    %xmm2, %xmm14
        addps   %xmm14, %xmm13
        movaps  %xmm3, %xmm14
        roundps $3, %xmm0, %xmm0
        cvttps2dq       %xmm0, %xmm0
        andps   %xmm1, %xmm14
        paddd   %xmm8, %xmm0
        orps    %xmm2, %xmm14
        pmaxsd  %xmm7, %xmm0
        addps   %xmm14, %xmm1
        movaps  %xmm3, %xmm14
        roundps $3, %xmm13, %xmm13
        cvttps2dq       %xmm13, %xmm13
        andps   %xmm12, %xmm14
        paddd   %xmm8, %xmm13
        pminsd  %xmm6, %xmm0
        orps    %xmm2, %xmm14
        pmaxsd  %xmm7, %xmm13
        pand    %xmm5, %xmm0
        addps   %xmm14, %xmm12
        roundps $3, %xmm1, %xmm1
        cvttps2dq       %xmm1, %xmm1
        paddd   %xmm8, %xmm1
        pmaxsd  %xmm7, %xmm1
        pminsd  %xmm6, %xmm13
        pminsd  %xmm6, %xmm1
        pand    %xmm5, %xmm13
        roundps $3, %xmm12, %xmm12
        cvttps2dq       %xmm12, %xmm12
        pand    %xmm5, %xmm1
        paddd   %xmm8, %xmm12
        pmaxsd  %xmm7, %xmm12
        packusdw        %xmm13, %xmm0
        pminsd  %xmm6, %xmm12
        pand    %xmm9, %xmm0
        pand    %xmm5, %xmm12
        packusdw        %xmm12, %xmm1
        pand    %xmm9, %xmm1
        packuswb        %xmm1, %xmm0
        movups  %xmm0, -16(%rdx)
        cmpq    %rax, %rcx
        jne     .L6
        movq    %rsi, %rax
        andq    $-16, %rax
        testb   $15, %sil
        je      .L1
        movq    %rsi, %rdx
        subq    %rax, %rdx
        leaq    -1(%rdx), %rcx
        cmpq    $6, %rcx
        jbe     .L8
.L5:
        leaq    (%rdi,%rax,4), %rcx
        pshufd  $0xe0, %xmm11, %xmm6
        pxor    %xmm8, %xmm8
        movq    .LC7(%rip), %xmm7
        movq    (%rcx), %xmm0
        movq    8(%rcx), %xmm5
        movq    16(%rcx), %xmm1
        movq    24(%rcx), %xmm11
        movq    %rdx, %rcx
        mulps   %xmm4, %xmm0
        andq    $-8, %rcx
        mulps   %xmm4, %xmm5
        mulps   %xmm4, %xmm1
        mulps   %xmm4, %xmm11
        movaps  %xmm3, %xmm4
        andps   %xmm0, %xmm4
        orps    %xmm2, %xmm4
        addps   %xmm4, %xmm0
        movaps  %xmm3, %xmm4
        andps   %xmm5, %xmm4
        orps    %xmm2, %xmm4
        addps   %xmm4, %xmm5
        movaps  %xmm3, %xmm4
        andps   %xmm11, %xmm3
        roundps $3, %xmm0, %xmm0
        andps   %xmm1, %xmm4
        cvttps2dq       %xmm0, %xmm0
        paddd   %xmm6, %xmm0
        orps    %xmm2, %xmm4
        orps    %xmm3, %xmm2
        addps   %xmm2, %xmm11
        addps   %xmm4, %xmm1
        roundps $3, %xmm5, %xmm5
        cvttps2dq       %xmm5, %xmm2
        paddd   %xmm6, %xmm2
        pmaxsd  %xmm8, %xmm0
        movq    .LC8(%rip), %xmm4
        pminsd  %xmm7, %xmm0
        roundps $3, %xmm1, %xmm1
        roundps $3, %xmm11, %xmm3
        cvttps2dq       %xmm1, %xmm1
        paddd   %xmm6, %xmm1
        cvttps2dq       %xmm3, %xmm11
        paddd   %xmm6, %xmm11
        pmaxsd  %xmm8, %xmm2
        pand    %xmm4, %xmm0
        pminsd  %xmm7, %xmm2
        pmaxsd  %xmm8, %xmm1
        pand    %xmm4, %xmm2
        pmaxsd  %xmm8, %xmm11
        pminsd  %xmm7, %xmm1
        packusdw        %xmm2, %xmm0
        movq    .LC9(%rip), %xmm2
        pminsd  %xmm7, %xmm11
        pand    %xmm4, %xmm1
        pshufd  $8, %xmm0, %xmm0
        pand    %xmm11, %xmm4
        pand    %xmm2, %xmm0
        packusdw        %xmm4, %xmm1
        pshufd  $8, %xmm1, %xmm1
        pand    %xmm1, %xmm2
        packuswb        %xmm2, %xmm0
        pshufd  $8, %xmm0, %xmm0
        movq    %xmm0, (%r8,%rax)
        addq    %rcx, %rax
        andl    $7, %edx
        je      .L1
.L8:
        movss   (%rdi,%rax,4), %xmm0
        movss   .LC10(%rip), %xmm2
        movss   .LC11(%rip), %xmm1
        leaq    0(,%rax,4), %rcx
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, (%r8,%rax)
        leaq    1(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        movss   4(%rdi,%rcx), %xmm0
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 1(%r8,%rax)
        leaq    2(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        movss   8(%rdi,%rcx), %xmm0
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 2(%r8,%rax)
        leaq    3(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        movss   12(%rdi,%rcx), %xmm0
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 3(%r8,%rax)
        leaq    4(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        movss   16(%rdi,%rcx), %xmm0
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 4(%r8,%rax)
        leaq    5(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        movss   20(%rdi,%rcx), %xmm0
        movaps  %xmm2, %xmm3
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm3
        orps    %xmm1, %xmm3
        addss   %xmm3, %xmm0
        pxor    %xmm3, %xmm3
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm3, %xmm0
        movdqa  .LC13(%rip), %xmm3
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 5(%r8,%rax)
        leaq    6(%rax), %rdx
        cmpq    %rsi, %rdx
        jnb     .L1
        mulss   24(%rdi,%rcx), %xmm15
        andps   %xmm15, %xmm2
        orps    %xmm2, %xmm1
        addss   %xmm1, %xmm15
        pxor    %xmm1, %xmm1
        roundss $3, %xmm15, %xmm15
        cvttss2sil      %xmm15, %edx
        movd    %edx, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm1, %xmm0
        pminsd  %xmm3, %xmm0
        movd    %xmm0, %edx
        movb    %dl, 6(%r8,%rax)
        ret
.L13:
        xorl    %edx, %edx
        movss   .LC10(%rip), %xmm3
        movss   .LC11(%rip), %xmm2
.L10:
        movss   (%rdi,%rdx,4), %xmm0
        movaps  %xmm3, %xmm1
        mulss   %xmm15, %xmm0
        andps   %xmm0, %xmm1
        orps    %xmm2, %xmm1
        addss   %xmm1, %xmm0
        pxor    %xmm1, %xmm1
        roundss $3, %xmm0, %xmm0
        cvttss2sil      %xmm0, %eax
        movd    %eax, %xmm0
        paddd   %xmm10, %xmm0
        pmaxsd  %xmm1, %xmm0
        movdqa  .LC13(%rip), %xmm1
        pminsd  %xmm1, %xmm0
        movd    %xmm0, %eax
        movb    %al, (%r8,%rdx)
        addq    $1, %rdx
        cmpq    %rdx, %rsi
        jne     .L10
.L1:
        ret
.L14:
        movss   .LC1(%rip), %xmm2
        movss   .LC3(%rip), %xmm3
        movq    %rsi, %rdx
        xorl    %eax, %eax
        movaps  %xmm15, %xmm4
        shufps  $0, %xmm4, %xmm4
        shufps  $0, %xmm2, %xmm2
        shufps  $0, %xmm3, %xmm3
        jmp     .L5
.LC7:
        .long   255
        .long   255
.LC8:
        .long   65535
        .long   65535
.LC9:
        .value  255
        .value  255
        .value  255
        .value  255
.LC10:
        .long   -2147483648
        .long   0
        .long   0
        .long   0
.LC11:
        .long   1056964607
        .long   0
        .long   0
        .long   0
.LC13:
        .long   255
        .long   0
        .long   0
        .long   0