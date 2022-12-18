#include <vector>
#include "datahandler.h"
#include "../../include/lodepng/lodepng.h"

/*
Функция считывает изображение в вектор char и переводит в 3D массив с 3 каналами RGB типов int.
[каналы, высота, ширина]
*/
void ReadImage(const char* path, Matrix3D& output, unsigned int& width, unsigned int& height)
{

    std::vector<unsigned char> image;

    // const char* path = "/home/galahad/Documents/5_course/1_semester/SRW/data/mask_full.png";
    unsigned error = lodepng::decode(image, width, height, path);

    // if(error) 
    // {
    //     std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    // }

    output.resize(height, std::vector<std::vector<float>>(width, std::vector<float>(3)));


    for (int i = 0; i < (int)height; i++)
    {
        for (int j = 0; j < (int)width; j++)
        {
            int index_1D = i*width*4 + j*4;

            output[i][j][0] = (int)image[index_1D];
            output[i][j][1] = (int)image[index_1D + 1];
            output[i][j][2] = (int)image[index_1D + 2];
        }
    }
}

/*
Функция переводит 3D масив в вектор типов char с 4 каналами RGBA и записывает в файл.
*/
void WriteImage(const char* path, const Matrix3D& image3d ,unsigned int& width, unsigned int& height)
{
    std::vector<unsigned char> output = std::vector<unsigned char>(width*height*4);
    for (int i = 0; i < (int)height; i++)
    {
        for (int j = 0; j < (int)width; j++)
        {
            int index_1D = i*width*4 + j*4;

            output[index_1D] = (char)image3d[i][j][0];
            output[index_1D + 1] = (char)image3d[i][j][1];
            output[index_1D + 2] = (char)image3d[i][j][2];
            output[index_1D + 3] = (char)255;
        }
    }

    unsigned int error = lodepng::encode(path, output, width, height);

    // if(error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}