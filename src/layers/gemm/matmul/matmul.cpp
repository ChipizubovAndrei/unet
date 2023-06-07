#include <immintrin.h>
#include <algorithm>

#include "matmul.h"

void gemm_native(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K)
{
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

// template <typename Dtype>
void gemm_nn(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc)
{
    int i,j,k;
    // #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// template <typename Dtype>
void gemm_nt(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc)
{
    int i,j,k;
    // #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

// template <typename Dtype>
void gemm_tn(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc)
{
    int i,j,k;
    // #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// template <typename Dtype>
void gemm_tt(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc)
{
    int i,j,k;
    // #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

// template <typename Dtype>
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float BETA,
                float *C, int ldc) {
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

// template <typename Dfloat
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
            float *A, int lda,
            float *B, int ldb,
            float BETA,
            float *C, int ldc) {
    gemm_cpu( TA,  TB,  M, N, K, ALPHA, A, lda, 
                B, ldb, BETA, C, ldc);
}