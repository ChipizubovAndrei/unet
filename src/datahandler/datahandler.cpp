#include "datahandler.h"
#include "../../include/lodepng/lodepng.h"

/*
Функция считывает изображение в вектор char и переводит в 1D массив с 3 каналами RGB типов int.
[каналы, высота, ширина]
Аргументы:
- path - путь к изображению
- output - возвращаемый массив
- srcH - возвращаемое значание высоты изображения
- srcW - возврщаемое значение ширины изображения
*/
void ReadImage(const char* path, float*& output, unsigned int& srcH, unsigned int& srcW)
{
    // Количество выходных каналов
    int srcC = 3;

    // std::vector<unsigned char> image;
    unsigned char* image = 0;

    unsigned error = lodepng_decode32_file(&image, &srcW, &srcH, path);

    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    output = new float [srcH*srcW*srcC];
    float* buf = output;

    for (int i = 0; i < (int)srcH; i++)
    {
        for (int j = 0; j < (int)srcW; j++)
        {
            *buf++ = (int)*image++;
            *buf++ = (int)*image++;
            *buf++ = (int)*image++;
            image++;
        }
    }
}

/*
Функция переводит 3D масив в вектор типов char с 4 каналами RGBA и записывает в файл.
Аргументы:
- path - путь к изображению
- src - входное изображение
- srcH - высота изображения
- srcW - ширина изображения
*/
void WriteImage(const char* path, float* src ,unsigned int& srcH, unsigned int& srcW)
{
    // Количество каналов в png
    int pngC = 4;
    unsigned char* output = new unsigned char [srcH*srcW*pngC];
    unsigned char* buf = output;

    for (int i = 0; i < (int)srcH; i++)
    {
        for (int j = 0; j < (int)srcW; j++)
        {
            *buf++ = (char)*src++;
            *buf++ = (char)*src++;
            *buf++ = (char)*src++;
            *buf++ = (char)255;
        }
    }

    unsigned int error = lodepng_encode32_file(path, output, srcW, srcH);

    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}