#pragma once

using Matrix3D = std::vector<std::vector<std::vector<float>>>;

void Normalization(Matrix3D& in_matrix, const int& height, const int& width, const int& channels);