#include <vector>
#include <cmath>
#include "Activations.h"


void ReLU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                if (in_matrix[y][x][c] < 0)
                {
                    in_matrix[y][x][c] = 0;
                }
            }
        }
    }
}

void LeakyReLU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels, float negative_slope)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                if (in_matrix[y][x][c] < 0)
                {
                    in_matrix[y][x][c] = negative_slope * in_matrix[y][x][c];
                }
            }
        }
    }
}

void ELU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels, float alpha)
{
    float E = 3.1415;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                if (in_matrix[y][x][c] < 0)
                {
                    in_matrix[y][x][c] = alpha * (pow(E, in_matrix[y][x][c]) - 1);
                }
            }
        }
    }
}

Matrix3D Softmax(Matrix3D& in_matrix, const int& height, const int& width, const int& in_channels)
{
    int segments[5][3] = {{0, 0, 0}, {0, 0, 255}, {0, 255, 255}, {255, 0, 255}, {255, 255, 0}};
    int max_ch_val;
    Matrix3D out_matrix(height, std::vector<std::vector<float>>(width, std::vector<float>(3)));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int ch = 0; ch < in_channels; ch++)
            {
                if (ch == 0)
                {
                    max_ch_val = 0;
                }
                else
                {
                    if (in_matrix[y][x][ch] > in_matrix[y][x][max_ch_val])
                    {
                        max_ch_val = ch;
                    }
                }
            }
            out_matrix[y][x][0] = segments[max_ch_val][0];
            out_matrix[y][x][1] = segments[max_ch_val][1];
            out_matrix[y][x][2] = segments[max_ch_val][2];
        }
    }
    return out_matrix;
}