
/*
Функция стандартного матричного умножения
Аргументы:
    - buf - входное изображение в формате dstH*dstW*kernelY*kernelX*srcC
    - weights - массив весов
    - bias - массив смещений
    - dst - возвращаемый массив
    - M - высота входного массива (dstH*dstW)
    - K - ширина входного массива (высова массива весов)
    - N - ширина массива весов
*/
void gemm_native(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

/*
Функция стандартного матричного умножения
(упрощено вычисление адресов массивов и обращение к элементам массива B теперь последовательное)
Аргументы:
    - buf - входное изображение в формате dstH*dstW*kernelY*kernelX*srcC
    - weights - массив весов
    - bias - массив смещений
    - dst - возвращаемый массив
    - M - высота входного массива (dstH*dstW)
    - K - ширина входного массива (высова массива весов)
    - N - ширина массива весов
*/
void gemm_v1(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

/*
Функция стандартного матричного умножения
(упрощено вычисление адресов массивов и обращение к элементам массива B теперь последовательное)
Аргументы:
    - buf - входное изображение в формате dstH*dstW*kernelY*kernelX*srcC
    - weights - массив весов
    - bias - массив смещений
    - dst - возвращаемый массив
    - M - высота входного массива (dstH*dstW)
    - K - ширина входного массива (высова массива весов)
    - N - ширина массива весов
*/
void gemm_avx(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

void gemm_v3(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

void gemm_v4(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

void gemm_v5(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

void add_bias(const float * bias, float * dst, int& M, int& N);
