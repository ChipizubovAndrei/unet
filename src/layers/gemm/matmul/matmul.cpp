#include <immintrin.h>
#include <algorithm>

#include "matmul.h"

void gemm_native(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    #pragma omp parallel for 
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            dst[m*N + n] += bias[n];
            for (int k = 0; k < K; k++)
            {
                dst[m*N + n] += buf[m*K + k]*weight[k*N + n];
            }
        }
    }
}

void gemm_v1(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    #pragma omp parallel for 
    for (int m = 0; m < M; m++)
    {
        float* c = dst + m * N;
        for (int n = 0; n < N; n++)
        {
            c[n] = bias[n];
        }
        for (int k = 0; k < K; ++k)
        {
            const float* b = weight + k * N;
            float a = buf[m * K + k];
            for (int n = 0; n < N; n++)
            {
                c[n] += a * b[n];
            }
        }
    }
}

void gemm_avx(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    if (N < 8)
    {
        gemm_v1(buf, weight, bias, dst, M, N, K);
    }
    else
    {
        #pragma omp parallel for 
        for (int i = 0; i < M; ++i)
        {
            float * c = dst + i * N;
            const float * bb = bias;
            for (int j = 0; j < N; j += 8)
                _mm256_storeu_ps(c + j + 0, _mm256_loadu_ps(bb + j + 0));
            for (int k = 0; k < K; ++k)
            {
                const float * b = weight + k * N;
                __m256 a = _mm256_set1_ps(buf[i*K + k]);
                for (int j = 0; j < N; j += 8)
                {
                    _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
                    // _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
                }
            }
        }
    }

}


/*---------------------------------------------------------------------------------------*/

void micro_4x16(int K, const float * A, int lda, int step, 
    const float * B, int ldb, float * C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    // __m256 c40 = _mm256_setzero_ps();
    // __m256 c50 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    // __m256 c41 = _mm256_setzero_ps();
    // __m256 c51 = _mm256_setzero_ps();
    const int offset0 = lda * 0;
    const int offset1 = lda * 1;
    const int offset2 = lda * 2;
    const int offset3 = lda * 3;
    // const int offset4 = lda * 4;
    // const int offset5 = lda * 5;
    __m256 b0, b1, a0, a1;
    for (int k = 0; k < K; k++)
    {
        b0 = _mm256_loadu_ps(B + 0);
        b1 = _mm256_loadu_ps(B + 8);
        a0 = _mm256_set1_ps(A[offset0]);
        a1 = _mm256_set1_ps(A[offset1]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        a0 = _mm256_set1_ps(A[offset2]);
        a1 = _mm256_set1_ps(A[offset3]);
        c20 = _mm256_fmadd_ps(a0, b0, c20);
        c21 = _mm256_fmadd_ps(a0, b1, c21);
        c30 = _mm256_fmadd_ps(a1, b0, c30);
        c31 = _mm256_fmadd_ps(a1, b1, c31);
        // a0 = _mm256_set1_ps(A[offset4]);
        // a1 = _mm256_set1_ps(A[offset5]);
        // c40 = _mm256_fmadd_ps(a0, b0, c40);
        // c41 = _mm256_fmadd_ps(a0, b1, c41);
        // c50 = _mm256_fmadd_ps(a1, b0, c50);
        // c51 = _mm256_fmadd_ps(a1, b1, c51);
        B += ldb; A += step;
    }
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
    C += ldc;
    // _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
    // _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
    // C += ldc;
    // _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
    // _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

void init_c(int M, int N, float * C, int ldc, const float* bias)
{
    const float * bb = bias;
    for (int i = 0; i < M; ++i, C += ldc)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_loadu_ps(bb + j + 0));
}

void gemm_v3(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    if (N < 8)
    {
        gemm_v1(buf, weight, bias, dst, M, N, K);
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < M; i += 4)
        {
            for (int j = 0; j < N; j += 16)
            {
                init_c(4, 16, dst + i*N + j, N, bias + j);
                micro_4x16(K, buf + i*K, K, 1, weight + j, N, dst + i*N + j, N);
            }
        }
    }
}

/*--------------------------------------------*/

struct buf_t
{
    float * p;
    int n;

    buf_t(int size) : n(size), p((float*)_mm_malloc(size * 4, 64)) {}
    ~buf_t() { _mm_free(p); }
};

void reorder_b_16(int K, const float * B, int ldb, float * bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void gemm_v4(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    if (N < 8)
    {
        gemm_v1(buf, weight, bias, dst, M, N, K);
    }
    else
    {
        #pragma omp parallel for
        for (int j = 0; j < N; j += 16)
        {
            buf_t bufB(16*K);
            reorder_b_16(K, weight + j, N, bufB.p);
            for (int i = 0; i < M; i += 4)
            {
                init_c(4, 16, dst + i*N + j, N, bias + j);
                micro_4x16(K, buf + i*K, K, 1, bufB.p, 16, dst + i*N + j, N);
            }
        }
    }
}

/*---------------------------------------------------*/

// void init_c_v5(int M, int N, float * C, int ldc)
// {
//     for (int i = 0; i < M; ++i, C += ldc)
//         for (int j = 0; j < N; j += 8)
//             _mm256_storeu_ps(C + j, _mm256_setzero_ps());
// }

void macro_v5(int M, int N, int K, const float * A, int lda, 
    const float * B, int ldb, float * bufB, float * C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        #pragma omp parallel for
        for (int i = 0; i < M; i += 4)
            micro_4x16(K, A + i*lda, lda, 1, bufB, 16, C + i*ldc + j, ldc);
    }
}

void gemm_v5(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
    if (N < 8){
        gemm_v1(buf, weight, bias, dst, M, N, K);
    }
    else{
        const int L1 = 32 * 1024;
        int mK = std::min(L1 / 4 / 16, K);
        buf_t bufB(16 * mK);
        for( int k = 0; k < K; k += mK ){
            int dK = std::min(K, k + mK) - k;
            if(k == 0)
                // init_c_v5(M, N, dst, N);
                add_bias(bias, dst, M, N);
            macro_v5(M, N, dK, buf + k, K, weight + k*N, N, bufB.p, dst, N);
        }
    }
}

void add_bias(const float * bias, float * dst, int& M, int& N)
{
    for (int i = 0; i < M; ++i, dst += N)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(dst + j, _mm256_loadu_ps(bias + j + 0));
    
    // #pragma omp parallel for
    // for (int m = 0; m < M; m++)
    // {
    //     float* c = dst + m * N;
    //     for (int n = 0; n < N; n++)
    //     {
    //         c[n] = bias[n];
    //     }
    // }
}

/*--------------------------------------------------------------*/