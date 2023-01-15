#pragma once
/*
Класс светрки
*/
class Convolution
{
private:

    int m_srcC; // Количество входных каналов
    int m_dstC; // Количество выходных каналов
    int m_kernekYX; // Размер ядра свертки
    int m_pad = 0; // Паддинг
    int m_stride = 1; // Страйд
    int m_dilation = 1; // Расстояние между элементами ядра

    char m_path_p1 [15] = "model_weights/";
    char m_path_kernel_p3 [10] = "/kernel:0";
    char m_path_bias_p3 [8] = "/bias:0";
    // Матрица фильтров
    float* m_weights;
    // Матрица смещений
    float* m_bias;

public:

    /*
    Аргументы:
        - path_p2 - путь к весам фильтров
        - in_channels - количество входных каналов
        - out_channels - количество выходных каналов
        - kernel_size - размер ядра свертки
    */
    Convolution(char* path_p2, const int in_channels, 
                const int out_channels, const int kernel_size);

    /*
    Аргументы:
        - path_p2 - путь к весам фильтров
        - in_channels - количество входных каналов
        - out_channels - количество выходных каналов
        - kernel_size - размер ядра свертки
        - padding - паддинг
        - stride - смещение
    */
    Convolution(char* path_p2, const int in_channels, 
                const int out_channels, const int kernel_size, int const padding, 
                int const stride);
    
    ~Convolution();
    
    /*
    Функция 2D свертки
    Классическая свертка (возвращает новую матрицу)
    Аргументы:
        - src - выход с предыдущего слоя
        - srcH - высота матрицы выхода предыдущего слоя
        - srcW - ширина матрицы выхода предыдущего слоя
    */
    float* Convolution2D(float* src, unsigned int& srcH, 
                            unsigned int& srcW);

    /*
    Функция 2D свертки
    Светка с использованием GeMM (возвращает новую матрицу)
    Аргументы:
        - src - выход с предыдущего слоя
        - srcH - высота матрицы выхода предыдущего слоя
        - srcW - ширина матрицы выхода предыдущего слоя
    */
    float* Convolution2D_GeMM(float* src, unsigned int& srcH, 
                            unsigned int& srcW);
    
};