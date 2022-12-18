#pragma once

class Concatenation
{
private:
    typedef std::vector<float> Matrix1D;
    typedef std::vector<std::vector<float>> Matrix2D;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    typedef std::vector<std::vector<std::vector<std::vector <float>>>> Matrix4D;
    

    int m_in_channels; // Количество входных каналов

public:
    Concatenation(const int in_channels, const int out_channels)
    {
        m_in_channels = in_channels;
    };

    ~Concatenation()
    {
        // std::cout << "Destroy Concatenation Layer" << std::endl;
    };

    Matrix3D Concatenation2D(Matrix3D& up_layer_out, Matrix3D& conv_layer_out, unsigned int& in_height, unsigned int& in_width);
};

