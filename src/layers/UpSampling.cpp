#include "UpSampling.h"

float* UpSample::UpSample2D(float* src, unsigned int& srcH, unsigned int& srcW)
{
    int dstH = (int)(srcH*m_scale_factor);
    int dstW = (int)(srcW*m_scale_factor);

    float* dst = new float [dstH*dstW*m_srcC];
    // #pragma omp parallel for 
    for (int sy = 0; sy < (int)srcH; sy++)
    {
        for (int sx = 0; sx < (int)srcW; sx++)
        {
            float* psrc = src + (sy*srcW + sx)*m_srcC;
            for (int ky = 0; ky < m_scale_factor; ky++)
            {
                for (int kx = 0; kx < m_scale_factor; kx++)
                {
                    int dy = sy * m_scale_factor + ky;
                    int dx = sx * m_scale_factor + kx;
                    for (int sc = 0; sc < m_srcC; sc++)
                    {
                        dst[(dy*dstW + dx)*m_srcC + sc] = psrc[sc];
                    }
                }
            }
        }
    }
    srcH = dstH;
    srcW = dstW;
    delete [] src;
    return dst;
}