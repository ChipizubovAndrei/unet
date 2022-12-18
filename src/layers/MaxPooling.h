#pragma once
/*
Класс макс пуллинга
*/
class MaxPool
{
private:
    typedef std::vector<float> Matrix1D;
    typedef std::vector<std::vector<float>> Matrix2D;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    typedef std::vector<std::vector<std::vector<std::vector <float>>>> Matrix4D;

    int m_in_channels; // Количество входных каналов
    int m_kernel_size; // Размер ядра свертки
    int m_padding = 0; // Паддинг
    int m_stride = 1; // Смещение ядра во время свертки
    int m_dilation = 1; // Расстояние между элементами ядра

public:

    MaxPool(const int in_channels, const int kernel_size)
    {
        m_in_channels = in_channels;
        m_kernel_size = kernel_size;
    };

    MaxPool(const int in_channels, const int kernel_size, int const padding, int const stride)
    {
        m_in_channels = in_channels;
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
    - prev_layer_out - выход с предыдущего слоя
    - in_height - высота матрицы выхода предыдущего слоя
    - in_width - ширина матрицы выхода предыдущего слоя
    */
    Matrix3D MaxPool2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width);
};