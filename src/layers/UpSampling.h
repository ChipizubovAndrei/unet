#pragma once
/*
Класс апсэмплинга
*/
class UpSample
{
private:

    int m_srcC; // Количество входных каналов
    int m_scale_factor; // Множитель пространственного размера

public:

    UpSample(const int in_channels, int scale_factor)
    {
        m_srcC = in_channels;
        m_scale_factor = scale_factor;
    };
    
    ~UpSample()
    {
        // std::cout << "Destroy UpLayer" << std::endl;
    };

    /*
    Апсэмплинг
    Аргументы:
        - src - выходное изобрадение
        - srcH - высота входного изображения
        - srcW - ширина входного изображения
    */
    float* UpSample2D(float* src, unsigned int& srcH, unsigned int& srcW);
};