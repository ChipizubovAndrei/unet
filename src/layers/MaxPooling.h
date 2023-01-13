#pragma once
/*
Класс макс пуллинга
*/
class MaxPool
{
private:

    int m_srcC; // Количество входных каналов
    int m_kernel_size; // Размер ядра свертки
    int m_padding = 0; // Паддинг
    int m_stride = 1; // Смещение ядра во время свертки
    int m_dilation = 1; // Расстояние между элементами ядра

public:

    MaxPool(const int in_channels, const int kernel_size)
    {
        m_srcC = in_channels;
        m_kernel_size = kernel_size;
    };

    MaxPool(const int in_channels, const int kernel_size, int const padding, int const stride)
    {
        m_srcC = in_channels;
        m_kernel_size = kernel_size;
        m_padding = padding;
        m_stride = stride;
    };
    
    ~MaxPool()
    {
        // std::cout << "Destroy PoolLayer" << std::endl;
    };

    /*
    Функция 2D макс пуллинга (возвращает новую матрицу)
    - src - входное изображение
    - srcH - высота входного изображения
    - srcW - ширина входного изображения
    */
    float* MaxPool2D(float* src, unsigned int& srcH, unsigned int& srcW);
};