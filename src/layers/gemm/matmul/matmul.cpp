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
    for (int m = 0; m < M; m++)
    {
        float* c = dst + m * N;
        for (int n = 0; n < N; n++)
            c[n] = bias[n];
        for (int k = 0; k < K; ++k)
        {
            const float* b = weight + k * N;
            float a = buf[m * K + k];
            for (int n = 0; n < N; n++)
                c[n] += a * b[n];
        }
    }
}