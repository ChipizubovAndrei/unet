#include "Concatenation.h"

float* Concatenation::Concatenation2D(float* src, float* clo, unsigned int& srcH, unsigned int& srcW)
{
    int dstC = (int)(2*m_srcC);

    float* dst = new float [srcH*srcW*dstC];
    // #pragma omp parallel for 
    for (int sy = 0; sy < (int)srcH; sy++)
    {
        for (int sx = 0; sx < (int)srcW; sx++)
        {
            for (int sc = 0; sc < dstC; sc++)
            {
                if (sc < m_srcC)
                {
                    dst[(sy*srcW + sx)*dstC + sc] = clo[(sy*srcW + sx)*m_srcC + sc];
                }
                else
                {
                    dst[(sy*srcW + sx)*dstC + sc] = src[(sy*srcW + sx)*m_srcC + sc - m_srcC];
                }
            }
        }
    }
    delete [] clo;
    delete [] src;
    return dst;
}