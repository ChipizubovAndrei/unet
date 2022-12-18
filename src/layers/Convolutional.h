#pragma once
/*
Класс светрки
*/
class Convolution
{
private:
    typedef std::vector<float> Matrix1D;
    typedef std::vector<std::vector<float>> Matrix2D;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    typedef std::vector<std::vector<std::vector<std::vector <float>>>> Matrix4D;
    

    int m_in_channels; // Количество входных каналов
    int m_out_channels; // Количество выходных каналов
    int m_kernel_size; // Размер ядра свертки
    int m_padding = 0; // Паддинг
    int m_stride = 1; // Страйд
    int m_dilation = 1; // Расстояние между элементами ядра

    char m_path_p1 [15] = "model_weights/";
    char m_path_kernel_p3 [10] = "/kernel:0";
    char m_path_bias_p3 [8] = "/bias:0";
    // Матрица фильтров
    float* m_kernel;
    // Матрица смещений
    float* m_bias;

public:

    Convolution(char* path_p2, const int in_channels, const int out_channels, const int kernel_size);

    Convolution(char* path_p2, const int in_channels, const int out_channels, const int kernel_size, int const padding, int const stride);
    
    ~Convolution();
    
    /*
    Функция 2D свертки
    Классическая свертка (возвращает новую матрицу)
    - prev_layer_out - выход с предыдущего слоя
    - in_height - высота матрицы выхода предыдущего слоя
    - in_width - ширина матрицы выхода предыдущего слоя
    */
    Matrix3D Convolution2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width);
    
    /*
    Светка методом Шмуэля Винограда
    */
    Matrix3D FastConvolution2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width);
};