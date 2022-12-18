#include <vector>
#include <string.h>
#include "paramgetter/paramgetter.h"
#include "Convolutional.h"


Convolution::Convolution(char* path_p2, const int in_channels, const int out_channels, const int kernel_size)
{
    m_in_channels = in_channels;
    m_out_channels = out_channels;
    m_kernel_size = kernel_size;

    m_kernel = new float [m_kernel_size*m_kernel_size*m_in_channels*out_channels];
    m_bias = new float [m_out_channels];

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_kernel, m_out_channels, m_in_channels, m_kernel_size);
    ParameterGetter(path_bias, m_bias, m_out_channels, 0, 0);
};

Convolution::Convolution(char* path_p2, const int in_channels, const int out_channels, const int kernel_size, int const padding, int const stride)
{
    m_in_channels = in_channels;
    m_out_channels = out_channels;
    m_kernel_size = kernel_size;
    m_padding = padding;
    m_stride = stride;

    m_kernel = new float [m_kernel_size*m_kernel_size*m_in_channels*out_channels];
    m_bias = new float [m_out_channels];

    char path_kernel[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_kernel_p3) + 1] = "";
    char path_bias [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_bias_p3) + 1] = "";

    strcat(path_kernel, m_path_p1);
    strcat(path_kernel, path_p2);
    strcat(path_kernel, m_path_kernel_p3);

    strcat(path_bias, m_path_p1);
    strcat(path_bias, path_p2);
    strcat(path_bias, m_path_bias_p3);

    ParameterGetter(path_kernel, m_kernel, m_out_channels, m_in_channels, m_kernel_size);
    ParameterGetter(path_bias, m_bias, m_out_channels, 0, 0);
};

Convolution::~Convolution()
{
    delete [] m_kernel; // (размер ядра, входные каналы, выходные каналы)
    delete [] m_bias; // (выходные каналы)
};

Convolution::Matrix3D Convolution::Convolution2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width)
{
    int out_height = (int)((in_height + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);
    int out_width = (int)((in_width + 2 * m_padding - m_dilation * (m_kernel_size - 1) - 1) 
                            / m_stride + 1);

    Matrix3D out_matrix(out_height, Matrix2D(out_width, Matrix1D(m_out_channels)));
    
    // Перебор по фильтрам (выходным каналам)
    for (int out_ch = 0; out_ch < m_out_channels; out_ch++)
    {
        // Проход по изображению
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                // Проход по ядру
                // Сумма значений по одному фильтру (m_in_channel ядер) в одном пикселе выходной матрицы
                double filter_sum = 0;
                for (int i = 0; i < m_kernel_size; i++)
                {
                    for (int j = 0; j < m_kernel_size; j++)
                    {
                        for (int in_ch = 0; in_ch < m_in_channels; in_ch++)
                        {
                            int in_y = (int)(y*m_stride - m_padding + i);
                            int in_x = (int)(x*m_stride - m_padding + j);
                            if (in_y < 0 || in_x < 0 || in_y >= (int)in_height || in_x >= (int)in_width)
                            {
                                filter_sum += 0;
                            }
                            else
                            {
                                /*
                                Так как m_kernel это 4-х мерный массив преобразованный в одномерный, необходимо пересчитать координаты осей
                                */
                                int dim3 = in_ch*m_out_channels;
                                int dim2 = j*m_in_channels*m_out_channels;
                                int dim1 = i*m_kernel_size*m_in_channels*m_out_channels;

                                filter_sum += prev_layer_out[in_y][in_x][in_ch]*m_kernel[dim1 + dim2 + dim3 + out_ch];
                            }   
                        }
                    }
                }
                out_matrix[y][x][out_ch] = filter_sum + m_bias[out_ch];
            }
        }
    }
    // delete [] m_kernel;
    // delete [] m_bias;
    in_height = out_height;
    in_width = out_width;
    return out_matrix;
}