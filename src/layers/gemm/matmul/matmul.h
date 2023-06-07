
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

void gemm_avx_v3(const float* buf, const float* weight, 
    const float* bias, float* dst, int& M, int& N, int& K);

void gemm_avx_v4(int M, int N, int K, const float * A, const float * B, float * C);

void gemm_v5(int M, int N, int K, const float * A, const float * B, float * C);

// template <typename Dtype>
void gemm_nn(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc);


void gemm_nt(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc);

void gemm_tn(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc);

void gemm_tt(int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
                float *A, int lda,
                float *B, int ldb,
                float BETA,
                float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
            float *A, int lda,
            float *B, int ldb,
            float BETA,
            float *C, int ldc);
