#pragma once

using Matrix3D = std::vector<std::vector<std::vector<float>>>;

void ReLU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels);
void LeakyReLU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels, float negative_slope);
void ELU(Matrix3D& in_matrix, const int& height, const int& width, const int& channels, float alpha);
Matrix3D Softmax(Matrix3D& in_matrix, const int& height, const int& width, const int& in_channels);