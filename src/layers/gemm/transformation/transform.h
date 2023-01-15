/*
Функция перевода изображения из формата H*W*C в формат H*W*kernelY*kernelX*C
Аргументы:
    - src - входное изображение в формате H*W*C
    - srcC - количество каналов в входном изображении
    - srcH - высота входного изображения
    - srcW - ширина входного изображения
    - kernelY - высота ядра свертки
    - kernelX - ширина ядра свертки
    - stride - смещение
    - pad - паддинг
    - buf - возвращаемое изображение в формате H*W*kernelY*kernelX*C
*/
void im2row(const float * src, int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int stride, int pad, float * buf);