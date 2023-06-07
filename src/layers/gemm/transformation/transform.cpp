#include "transform.h"

void im2row(const float * src, int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int stride, int pad, float * buf)
{
    int dstH = (srcH + 2 * pad - kernelY) / stride + 1;
    int dstW = (srcW + 2 * pad - kernelX) / stride + 1;
    #pragma omp parallel for 
    for (int dy = 0; dy < dstH; dy++)
    {
        for (int dx = 0; dx < dstW; dx++)
        {
            for (int ky = 0; ky < kernelY; ky++)
            {
                for (int kx = 0; kx < kernelX; kx++)
                {
                    int sy = dy * stride + ky - pad;
                    int sx = dx * stride + kx - pad;
                    for (int sc = 0; sc < srcC; sc++)
                    {
                        if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
                            *buf++ = src[(sy * srcW + sx)*srcC + sc];
                        else
                            *buf++ = 0;
                    }
                }
            }
        }
    }
}