#include <vector>
#include "Normalization.h"


void Normalization(Matrix3D& in_matrix, const int& height, const int& width, const int& channels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                in_matrix[y][x][c] = in_matrix[y][x][c] / 255;
            }
        }
    }
}