#include <string.h>
#include "gemm/matmul/matmul.h"
#include "gemm/transformation/transform.h"
#include "paramgetter/paramgetter.h"
#include "Convolutional.h"

// #include "gemm/matmul/cpu_calc_ops.h"


Convolution::Convolution(char* path_p2, const int in_channels, 
                        const int out_channels, const int kernel_size)
{
    m_srcC = in_channels;
    m_dstC = out_channels;
    m_kernekYX = kernel_size;

    m_weights = new float [m_kernekYX*m_kernekYX*m_srcC*m_dstC];
    m_bias = new float [m_dstC];

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_weights, m_dstC, m_srcC, m_kernekYX);
    ParameterGetter(path_bias, m_bias, m_dstC, 0, 0);
};

Convolution::Convolution(char* path_p2, const int in_channels, 
                        const int out_channels, const int kernel_size, int const padding, 
                        int const stride)
{
    m_srcC = in_channels;
    m_dstC = out_channels;
    m_kernekYX = kernel_size;
    m_pad = padding;
    m_stride = stride;

    m_weights = new float [m_kernekYX*m_kernekYX*m_srcC*m_dstC];
    m_bias = new float [m_dstC];
    // m_weights = (float*)malloc(m_kernekYX*m_kernekYX*m_srcC*m_dstC*sizeof(float));
    // m_bias = (float*)malloc(m_dstC * sizeof(float));

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_weights, m_dstC, m_srcC, m_kernekYX);
    ParameterGetter(path_bias, m_bias, m_dstC, 0, 0);
};

Convolution::~Convolution()
{
    delete [] m_weights;
    delete [] m_bias;
};

float* Convolution::Convolution2D(float* src, 
                unsigned int& srcH, unsigned int& srcW)
{
    int dstH = (int)((srcH + 2 * m_pad - m_dilation * (m_kernekYX - 1) - 1) / m_stride + 1);
    int dstW = (int)((srcW + 2 * m_pad - m_dilation * (m_kernekYX - 1) - 1) / m_stride + 1);

    float* dst = new float [dstH*dstW*m_dstC];
    
    // Перебор по фильтрам (выходным каналам)
    for (int dc = 0; dc < m_dstC; dc++)
    {
        // Проход по изображению
        for (int dy = 0; dy < dstH; dy++)
        {
            for (int dx = 0; dx < dstW; dx++)
            {
                // Проход по ядру
                // Сумма значений по одному фильтру (m_in_channel ядер) в одном пикселе выходной матрицы
                double sum = 0;
                for (int sc = 0; sc < m_srcC; sc++)
                {
                    for (int ky = 0; ky < m_kernekYX; ky++)
                    {
                        for (int kx = 0; kx < m_kernekYX; kx++)
                        {
                            int sy = (int)(dy*m_stride - m_pad + ky);
                            int sx = (int)(dx*m_stride - m_pad + kx);
                            if (sy >= 0 && sy < (int)srcH && sx >= 0 && sx < (int)srcW)
                            {
                                sum += src[(sy*srcW + sx)*m_srcC + sc]*
                                        m_weights[((ky*m_kernekYX + kx)*m_srcC + sc)*m_dstC + dc];
                            }   
                        }
                    }
                }
                dst[(dy*dstW + dx)*m_dstC + dc] = sum + m_bias[dc];
            }
        }
    }
    srcH = dstH;
    srcW = dstW;
    delete [] src;
    return dst;
}

float* Convolution::Convolution2D_GeMM(float* src, unsigned int& srcH, 
                            unsigned int& srcW)
{
    int dstH = (srcH + 2 * m_pad - m_kernekYX) / m_stride + 1;
    int dstW = (srcW + 2 * m_pad - m_kernekYX) / m_stride + 1;
    int M = dstH * dstW;
    int N = m_dstC;
    int K = m_srcC * m_kernekYX * m_kernekYX;

    float* buf = new float [M*K];
    float* dst = new float [dstH*dstW*m_dstC];
    // float alpha = 1.0;

    im2row(src, m_srcC, srcH, srcW, m_kernekYX, m_kernekYX, m_stride, m_pad, buf);
    gemm_v5(buf, m_weights, m_bias, dst, M, N, K);
    
    delete [] buf;
    return dst;
}