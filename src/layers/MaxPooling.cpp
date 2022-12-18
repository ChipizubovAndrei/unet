#include <vector>
#include "MaxPooling.h"


MaxPool::Matrix3D MaxPool::MaxPool2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width)
{
    int out_height = (int)((in_height + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);
    int out_width = (int)((in_width + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);

    // Копируем входную матрицу
    Matrix3D out_matrix(out_height, Matrix2D(out_width, Matrix1D(m_in_channels)));
    
    // Проход по изображению
    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            for (int i = 0; i < m_kernel_size; i++)
            {
                for (int j = 0; j < m_kernel_size; j++)
                {
                    for (int in_ch = 0; in_ch < m_in_channels; in_ch++)
                    {
                        float value = prev_layer_out[y*m_stride+i][x*m_stride+j][in_ch];
                        if (i == 0 && j == 0)
                        {
                            out_matrix[y][x][in_ch] = value;
                        }
                        else
                        {
                            if (out_matrix[y][x][in_ch] < value)
                            {
                                out_matrix[y][x][in_ch] = value;
                            }
                        }
                    }
                }
            }
        }
    }
    in_height = out_height;
    in_width = out_width;
    return out_matrix;
}