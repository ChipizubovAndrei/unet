#include "Normalization.h"

/*
Функция нормализации входного изображения
Аргументы:
- src - возвращаемое нормализованное изображение
- srcH - высота изображения
- srcW - ширина изображения
- srcC - количество каналов в изображениии
*/
void Normalization(float* src, const int& srcH, const int& srcW, const int& srcC)
{
    for (int ld = 0; ld < srcH*srcW*srcC; ld++)
    {
        *src++ = *src / 255;
    }

}