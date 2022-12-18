#include <vector>
#include "Concatenation.h"

Concatenation::Matrix3D Concatenation::Concatenation2D(Matrix3D& up_layer_out, Matrix3D& conv_layer_out, unsigned int& in_height, unsigned int& in_width)
{
    int out_channels = (int)(2*m_in_channels);

    Matrix3D out_matrix(in_height, Matrix2D(in_width, Matrix1D(out_channels)));
    
    // Перебор по фильтрам (выходным каналам)
    for (int y = 0; y < (int)in_height; y++)
    {
        // Проход по изображению
        for (int x = 0; x < (int)in_width; x++)
        {
            for (int ch = 0; ch < out_channels; ch++)
            {
                if (ch < m_in_channels)
                {
                    out_matrix[y][x][ch] = conv_layer_out[y][x][ch];
                }
                else
                {
                    out_matrix[y][x][ch] = up_layer_out[y][x][ch - m_in_channels];
                }
            }
        }
    }
    return out_matrix;
}