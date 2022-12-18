#pragma once
/*
Класс апсэмплинга
*/
class UpSample
{
private:
    typedef std::vector<float> Matrix1D;
    typedef std::vector<std::vector<float>> Matrix2D;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    typedef std::vector<std::vector<std::vector<std::vector <float>>>> Matrix4D;

    int m_in_channels; // Количество входных каналов
    int m_scale_factor; // Множитель пространственного размера

public:

    UpSample(const int in_channels, int scale_factor)
    {
        m_in_channels = in_channels;
        m_scale_factor = scale_factor;
    };
    
    ~UpSample()
    {
        // std::cout << "Destroy UpLayer" << std::endl;
    };

    /*Апсэмплинг
    - prev_layer_out - выход с предыдущего слоя
    - in_height - высота матрицы выхода предыдущего слоя
    - in_width - ширина матрицы выхода предыдущего слоя
    */
    Matrix3D UpSample2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width);
};