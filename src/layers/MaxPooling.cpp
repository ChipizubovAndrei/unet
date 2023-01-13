#include "MaxPooling.h"


float* MaxPool::MaxPool2D(float* src, unsigned int& srcH, unsigned int& srcW)
{
    int dstH = (int)((srcH + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);
    int dstW = (int)((srcW + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);

    // Копируем входную матрицу
    float* dst = new float [dstH*dstW*m_srcC];
    
    // Проход по изображению
    for (int dy = 0; dy < dstH; dy++)
    {
        for (int dx = 0; dx < dstW; dx++)
        {
            for (int ky = 0; ky < m_kernel_size; ky++)
            {
                for (int kx = 0; kx < m_kernel_size; kx++)
                {
                    for (int sc = 0; sc < m_srcC; sc++)
                    {
                        int sy = dy*m_stride + ky;
                        int sx = dx*m_stride + kx;
                        float value = src[(sy*srcH + sx)*m_srcC + sc];
                        if (ky == 0 && kx == 0)
                        {
                            dst[(dy*dstW + dx)*m_srcC + sc] = value;
                        }
                        else
                        {
                            if (dst[(dy*dstW + dx)*m_srcC + sc] < value)
                            {
                                dst[(dy*dstW + dx)*m_srcC + sc] = value;
                            }
                        }
                    }
                }
            }
        }
    }
    srcH = dstH;
    srcW = dstW;
    return dst;
}