#include "UpSampling.h"


float* UpSample::UpSample2D(float* src, unsigned int& srcH, unsigned int& srcW)
{
    int dstH = (int)(srcH*m_scale_factor);
    int dstW = (int)(srcW*m_scale_factor);

    float* dst = new float [dstH*dstW*m_in_channels];
    
    for (int sy = 0; sy < (int)srcH; sy++)
    {
        for (int sx = 0; sx < (int)srcW; sx++)
        {
            for (int sc = 0; sc < m_in_channels; sc++)
            {
                float value = src[(sy*srcW + sx)*m_in_channels + sc];
                for (int ky = 0; ky < m_scale_factor; ky++)
                {
                    for (int kx = 0; kx < m_scale_factor; kx++)
                    {
                        int dy = sy * m_scale_factor + ky;
                        int dx = sx * m_scale_factor + kx;
                        dst[(dy*dstW + dx)*m_in_channels + sc] = value;
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