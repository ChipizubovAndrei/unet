#pragma once

class Concatenation
{
private:
    
    int m_srcC; // Количество входных каналов

public:
    Concatenation(const int in_channels, const int out_channels)
    {
        m_srcC = in_channels;
    };

    ~Concatenation()
    {
        // std::cout << "Destroy Concatenation Layer" << std::endl;
    };

    /*
    Функция конкатенации
    Аргументы:
        - src - выход из предыдущего слоя нейронной сети
        - clo - выход сверточного слоя нейронной сети
        - srcH - высота изображения
        - srcW - ширина изображения
    */
    float* Concatenation2D(float* src, float* clo, unsigned int& srcH, unsigned int& srcW);
};

