#include "gemm/transformation/transform.h"
#include "MaxPooling.h"


float* MaxPool::MaxPool2D(float* src, unsigned int& srcH, unsigned int& srcW)
{
    int dstH = (int)((srcH + 2 * m_padding - m_dilation * (m_kernelYX - 1) - 1) 
                            / m_stride + 1);
    int dstW = (int)((srcW + 2 * m_padding - m_dilation * (m_kernelYX - 1) - 1) 
                            / m_stride + 1);

    int M = dstH * dstW;
    int K = m_kernelYX * m_kernelYX;

    float* dst = new float [dstH*dstW*m_srcC];
    float* buf = new float [M*K*m_srcC];
    
    im2row(src, m_srcC, srcH, srcW, m_kernelYX, m_kernelYX, m_stride, 0, buf);
    #pragma omp parallel for 
    for (int m = 0; m < M; m++)
    {
        float* pdst = dst + m*m_srcC;
        for (int k = 0; k < m_kernelYX*m_kernelYX; k++)
        {
            float* pbuf = buf + (m*K + k)*m_srcC;
            for (int n = 0; n < m_srcC; n++)
            {
                if (k == 0)
                {
                    pdst[n] = pbuf[n];
                }
                else if (pbuf[n] > pdst[n])
                {
                    pdst[n] = pbuf[n];
                }
            }
        }
    }
    srcH = dstH;
    srcW = dstW;
    return dst;
}