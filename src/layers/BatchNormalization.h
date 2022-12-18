#pragma once

class BatchNormalization
{
private:
    typedef std::vector<float> Matrix1D;
    typedef std::vector<std::vector<float>> Matrix2D;
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    typedef std::vector<std::vector<std::vector<std::vector <float>>>> Matrix4D;

    char m_path_p1 [15] = "model_weights/";
    char m_path_beta_p3 [8] = "/beta:0";
    char m_path_gamma_p3 [9] = "/gamma:0";
    char m_path_mean_p3 [15] = "/moving_mean:0";
    char m_path_var_p3 [19] = "/moving_variance:0";

    int m_in_channels; // Количество входных каналов
    float* m_beta; // Параметры сдвига
    float* m_gamma; // Парамертры маштабирования
    float* m_running_mean; // EMA среднего отклонения
    float* m_running_var; // EMA дисперсии

public:

    BatchNormalization(char* path_p2, const int in_channels);
    
    ~BatchNormalization();

    /*
    Функция 2D батч-нормализации (изменяет входную матрицу)
    - prev_layer_out - выход с предыдущего слоя
    - in_height - высота матрицы выхода предыдущего слоя
    - in_width - ширина матрицы выхода предыдущего слоя
    */
    void BatchNormalization2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width);
};