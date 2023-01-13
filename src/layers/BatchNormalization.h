#pragma once

class BatchNormalization
{
private:

    char m_path_p1 [15] = "model_weights/";
    char m_path_beta_p3 [8] = "/beta:0";
    char m_path_gamma_p3 [9] = "/gamma:0";
    char m_path_mean_p3 [15] = "/moving_mean:0";
    char m_path_var_p3 [19] = "/moving_variance:0";

    int m_srcC; // Количество входных каналов
    float* m_beta; // Параметры сдвига
    float* m_gamma; // Парамертры маштабирования
    float* m_running_mean; // EMA среднего отклонения
    float* m_running_var; // EMA дисперсии

public:

    BatchNormalization(char* path_p2, const int srcC);
    
    ~BatchNormalization();

    /*
    Функция 2D батч-нормализации (изменяет входную матрицу)
    Аргументы:
        - src - выход с предыдущего слоя
        - srcH - высота матрицы выхода предыдущего слоя
        - srcW - ширина матрицы выхода предыдущего слоя
    */
    void BatchNormalization2D(float* src, unsigned int& srcH, unsigned int& srcW);
};