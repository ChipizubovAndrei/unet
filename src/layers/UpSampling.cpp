#include <vector>
#include "UpSampling.h"


UpSample::Matrix3D UpSample::UpSample2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width)
{
    int out_height = (int)(in_height*m_scale_factor);
    int out_width = (int)(in_width*m_scale_factor);

    Matrix3D out_matrix(out_height, Matrix2D(out_width, Matrix1D(m_in_channels)));
    
    // Проход по изображению
    for (int y = 0; y < (int)in_height; y++)
    {
        for (int x = 0; x < (int)in_width; x++)
        {
            for (int in_ch = 0; in_ch < m_in_channels; in_ch++)
            {
                float value = prev_layer_out[y][x][in_ch];
                for (int i = 0; i < m_scale_factor; i++)
                {
                    for (int j = 0; j < m_scale_factor; j++)
                    {
                        out_matrix[y*m_scale_factor+i][x*m_scale_factor+j][in_ch] = value;
                    }
                }
            }
        }
    }
    in_height = out_height;
    in_width = out_width;
    return out_matrix;
}