#include <string.h>
#include "paramgetter/paramgetter.h"
#include "Convolutional.h"


Convolution::Convolution(char* path_p2, const int in_channels, 
                        const int out_channels, const int kernel_size)
{
    m_srcC = in_channels;
    m_dstC = out_channels;
    m_kernel_size = kernel_size;

    m_weights = new float [m_kernel_size*m_kernel_size*m_srcC*out_channels];
    m_bias = new float [m_dstC];

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_weights, m_dstC, m_srcC, m_kernel_size);
    ParameterGetter(path_bias, m_bias, m_dstC, 0, 0);
};

Convolution::Convolution(char* path_p2, const int in_channels, 
                        const int out_channels, const int kernel_size, int const padding, 
                        int const stride)
{
    m_srcC = in_channels;
    m_dstC = out_channels;
    m_kernel_size = kernel_size;
    m_padding = padding;
    m_stride = stride;

    m_weights = new float [m_kernel_size*m_kernel_size*m_srcC*out_channels];
    m_bias = new float [m_dstC];

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_weights, m_dstC, m_srcC, m_kernel_size);
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
    int dstH = (int)((srcH + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1);
    int dstW = (int)((srcW + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) / m_stride + 1);

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
                    for (int ky = 0; ky < m_kernel_size; ky++)
                    {
                        for (int kx = 0; kx < m_kernel_size; kx++)
                        {
                            int sy = (int)(dy*m_stride - m_padding + ky);
                            int sx = (int)(dx*m_stride - m_padding + kx);
                            if (sy >= 0 && sy < (int)srcH && sx >= 0 && sx < (int)srcW)
                            {
                                sum += src[(sy*srcW + sx)*m_srcC + sc]*
                                        m_weights[((ky*m_kernel_size + kx)*m_srcC + sc)*m_dstC + dc];
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