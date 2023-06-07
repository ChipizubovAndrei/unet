#include <cmath>
#include "Activations.h"

void ReLU(float* src, const int& srcH, const int& srcW, const int& srcC)
{
    // #pragma omp parallel for 
    for (int ld = 0; ld < srcH*srcW*srcC; ld++)
    {
        if (*src < 0)
        {
            *src = 0;
        }
        src++;
    }
}

void LeakyReLU(float* src, const int& srcH, const int& srcW, const int& srcC, float negative_slope)
{
    // #pragma omp parallel for 
    for (int ld = 0; ld < srcH*srcW*srcC; ld++)
    {
        if (*src < 0)
        {
            *src = negative_slope * (*src);
        }
        src++;
    }
}

void ELU(float* src, const int& srcH, const int& srcW, const int& srcC, float alpha)
{
    float E = 3.1415;
    // #pragma omp parallel for 
    for (int ld = 0; ld < srcH*srcW*srcC; ld++)
    {
        if (*src < 0)
        {
            *src = alpha * (pow(E, *src) - 1);
        }
        src++;
    }
}

float* Softmax(float* src, const int& srcH, const int& srcW, const int& srcC)
{
    // Количество выходных каналов
    int dstC = 3;
    int segments[5][3] = {{0, 0, 0}, {0, 0, 255}, {0, 255, 255}, {255, 0, 255}, {255, 255, 0}};
    int max_sc_val;
    float* dst = new float [srcH*srcW*dstC];
    // #pragma omp parallel for 
    for (int sy = 0; sy < srcH; sy++)
    {
        for (int sx = 0; sx < srcW; sx++)
        {
            for (int sc = 0; sc < srcC; sc++)
            {
                if (sc == 0)
                {
                    max_sc_val = 0;
                }
                else
                {
                    if (src[(sy*srcW + sx)*srcC + sc] > src[(sy*srcW + sx)*srcC + max_sc_val])
                    {
                        max_sc_val = sc;
                    }
                }
            }
            dst[(sy*srcW + sx)*dstC + 0] = segments[max_sc_val][0];
            dst[(sy*srcW + sx)*dstC + 1] = segments[max_sc_val][1];
            dst[(sy*srcW + sx)*dstC + 2] = segments[max_sc_val][2];
        }
    }
    delete [] src;
    return dst;
}