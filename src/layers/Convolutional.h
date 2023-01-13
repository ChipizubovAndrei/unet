#pragma once
/*
Класс светрки
*/
class Convolution
{
private:

    int m_srcC; // Количество входных каналов
    int m_dstC; // Количество выходных каналов
    int m_kernel_size; // Размер ядра свертки
    int m_padding = 0; // Паддинг
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

    Convolution(char* path_p2, const int in_channels, 
                const int out_channels, const int kernel_size);

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
    
};