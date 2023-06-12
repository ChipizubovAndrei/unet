#include <iostream>
#include <chrono>
#include <stdio.h>
#include "src/datahandler/datahandler.h"
#include "src/layers/Convolutional.h"
#include "src/layers/MaxPooling.h"
#include "src/layers/UpSampling.h"
#include "src/layers/BatchNormalization.h"
#include "src/layers/Concatenation.h"
#include "src/layers/Normalization.h"
#include "src/activation/Activations.h"

using namespace std;

// #define PRINT_BLOCK 

float* conv(float* src, unsigned int& srcH, unsigned int& srcW, 
            int srcC,  int dstC, char* conv_1, int kernel_size = 3, int pad = 1)
{
    Convolution conv_1_1(conv_1, srcC, dstC, kernel_size, pad, 1);
    src = conv_1_1.Convolution2D_GeMM(src, srcH, srcW);
    return src;
}

float* conv_block(float* src, unsigned int& srcH, unsigned int& srcW, int srcC,  int dstC,
            char* conv_1, char* batch_1, char* conv_2, char* batch_2)
{
    // Convolution conv_1_1(conv_1, srcC, dstC, 3, 1, 1);
    // src = conv_1_1.Convolution2D_GeMM(src, srcH, srcW);
    src = conv(src, srcH, srcW, srcC, dstC, conv_1);
    LeakyReLU(src, srcH, srcW, dstC, 0.3);
    BatchNormalization batch_1_1(batch_1, dstC);
    batch_1_1.BatchNormalization2D(src, srcH, srcW);

    // Convolution conv_1_2(conv_2, dstC, dstC, 3, 1, 1);
    // src = conv_1_2.Convolution2D_GeMM(src, srcH, srcW);
    src = conv(src, srcH, srcW, dstC, dstC, conv_2);
    LeakyReLU(src, srcH, srcW, dstC, 0.3);
    BatchNormalization batch_1_2(batch_2, dstC);
    batch_1_2.BatchNormalization2D(src, srcH, srcW);

    return src;
}

float* up_block(float* src, unsigned int& srcH, unsigned int& srcW, 
                int srcC,  int dstC,char* conv_1, char* batch_1)
{
    UpSample up = UpSample(srcC, 2);
    src = up.UpSample2D(src, srcH, srcW);

    Convolution conv(conv_1, srcC, dstC, 3, 1, 1);
    src = conv.Convolution2D_GeMM(src, srcH, srcW);
    LeakyReLU(src, srcH, srcW, dstC, 0.3);
    BatchNormalization batch(batch_1, dstC);
    batch.BatchNormalization2D(src, srcH, srcW);

    return src;
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    unsigned int height, width;
    
    float* original_image;
    float* output;
    float* output_blok_1;
    float* output_blok_2;
    float* output_blok_3;
    float* output_blok_4;
    float* output_image;

    char* conv_1 = "conv2d/conv2d";
    char* batch_1 = "batch_normalization/batch_normalization";
    char* conv_2 = "conv2d_1/conv2d_1";
    char* batch_2 = "batch_normalization_1/batch_normalization_1";

    char* conv_3 = "conv2d_2/conv2d_2";
    char* batch_3 = "batch_normalization_2/batch_normalization_2";
    char* conv_4 = "conv2d_3/conv2d_3";
    char* batch_4 = "batch_normalization_3/batch_normalization_3";

    char* conv_5 = "conv2d_4/conv2d_4";
    char* batch_5 = "batch_normalization_4/batch_normalization_4";
    char* conv_6 = "conv2d_5/conv2d_5";
    char* batch_6 = "batch_normalization_5/batch_normalization_5";

    char* conv_7 = "conv2d_6/conv2d_6";
    char* batch_7 = "batch_normalization_6/batch_normalization_6";
    char* conv_8 = "conv2d_7/conv2d_7";
    char* batch_8 = "batch_normalization_7/batch_normalization_7";

    char* conv_9 = "conv2d_8/conv2d_8";
    char* batch_9 = "batch_normalization_8/batch_normalization_8";
    char* conv_10 = "conv2d_9/conv2d_9";
    char* batch_10 = "batch_normalization_9/batch_normalization_9";

    char* conv_11 = "conv2d_10/conv2d_10";
    char* batch_11 = "batch_normalization_10/batch_normalization_10";
    char* conv_12 = "conv2d_11/conv2d_11";
    char* batch_12 = "batch_normalization_11/batch_normalization_11";

    char* conv_13 = "conv2d_12/conv2d_12";
    char* batch_13 = "batch_normalization_12/batch_normalization_12";
    char* conv_14 = "conv2d_13/conv2d_13";
    char* batch_14 = "batch_normalization_13/batch_normalization_13";

    char* conv_15 = "conv2d_14/conv2d_14";
    char* batch_15 = "batch_normalization_14/batch_normalization_14";
    char* conv_16 = "conv2d_15/conv2d_15";
    char* batch_16 = "batch_normalization_15/batch_normalization_15";

    char* conv_17 = "conv2d_16/conv2d_16";
    char* batch_17 = "batch_normalization_16/batch_normalization_16";
    char* conv_18 = "conv2d_17/conv2d_17";
    char* batch_18 = "batch_normalization_17/batch_normalization_17";

    char* conv_19 = "conv2d_18/conv2d_18";
    char* batch_19 = "batch_normalization_18/batch_normalization_18";
    char* conv_20 = "conv2d_19/conv2d_19";
    char* batch_20 = "batch_normalization_19/batch_normalization_19";

    char* conv_21 = "conv2d_20/conv2d_20";
    char* batch_21 = "batch_normalization_20/batch_normalization_20";
    char* conv_22 = "conv2d_21/conv2d_21";
    char* batch_22 = "batch_normalization_21/batch_normalization_21";

    char* conv_23 = "conv2d_22/conv2d_22";

    const char* path2image = "/home/galahad/Documents/5_course/1_semester/SRW/data/Abies Sibirica Dataset/Abies Sibirica Dataset/fragments/validation/x/875.png";
    const char* path2save = "/home/galahad/Documents/5_course/1_semester/SRW/unet_v_01/output/mask_875.png";

    ReadImage(path2image, original_image, width, height);

    Normalization(original_image, height, width, 3);
#ifdef PRINT_BLOCK
    cout << "Первый блок" << endl;
#endif // PRINT_BLOCK
    output_blok_1 = conv_block(original_image, height, width, 3, 64, conv_1, batch_1, conv_2, batch_2);
    MaxPool pool_1(64, 2, 0, 2);
    output = pool_1.MaxPool2D(output_blok_1, height, width);

#ifdef PRINT_BLOCK
    cout << "Второй блок" << endl;
#endif // PRINT_BLOCK
    output_blok_2 = conv_block(output, height, width, 64, 128, conv_3, batch_3, conv_4, batch_4);
    MaxPool pool_2(128, 2, 0, 2);
    output = pool_2.MaxPool2D(output_blok_2, height, width);

#ifdef PRINT_BLOCK
    cout << "Третий блок" << endl;
#endif // PRINT_BLOCK
    output_blok_3 = conv_block(output, height, width, 128, 256, conv_5, batch_5, conv_6, batch_6);
    MaxPool pool_3(256, 2, 0, 2);
    output = pool_3.MaxPool2D(output_blok_3, height, width);

#ifdef PRINT_BLOCK
    cout << "Четвертый блок" << endl;
#endif // PRINT_BLOCK
    output_blok_4 = conv_block(output, height, width, 256, 512, conv_7, batch_7, conv_8, batch_8);
    MaxPool pool_4(512, 2, 0, 2);
    output = pool_4.MaxPool2D(output_blok_4, height, width);

#ifdef PRINT_BLOCK
    cout << "Пятый блок" << endl;
#endif // PRINT_BLOCK
    output = conv_block(output, height, width, 512, 1024, conv_9, batch_9, conv_10, batch_10);
    output = up_block(output, height, width, 1024, 512, conv_11, batch_11);
    Concatenation merge_5 = Concatenation(512, 1024);
    output = merge_5.Concatenation2D(output, output_blok_4, height, width);

#ifdef PRINT_BLOCK
    cout << "Шестой блок" << endl;
#endif // PRINT_BLOCK
    output = conv_block(output, height, width, 1024, 512, conv_12, batch_12, conv_13, batch_13);
    output = up_block(output, height, width, 512, 256, conv_14, batch_14);
    Concatenation merge_6 = Concatenation(256, 512);
    output = merge_6.Concatenation2D(output, output_blok_3, height, width);

#ifdef PRINT_BLOCK
    cout << "Седьмой блок" << endl;
#endif // PRINT_BLOCK
    output = conv_block(output, height, width, 512, 256, conv_15, batch_15, conv_16, batch_16);
    output = up_block(output, height, width, 256, 128, conv_17, batch_17);
    Concatenation merge_7 = Concatenation(128, 256);
    output = merge_7.Concatenation2D(output, output_blok_2, height, width);

#ifdef PRINT_BLOCK
    cout << "Восьмой блок" << endl;
#endif // PRINT_BLOCK
    output = conv_block(output, height, width, 256, 128, conv_18, batch_18, conv_19, batch_19);
    output = up_block(output, height, width, 128, 64, conv_20, batch_20);
    Concatenation merge_8 = Concatenation(64, 128);
    output = merge_8.Concatenation2D(output, output_blok_1, height, width);

#ifdef PRINT_BLOCK    
    cout << "Девятый блок" << endl;
#endif // PRINT_BLOCK
    output = conv_block(output, height, width, 128, 64, conv_21, batch_21, conv_22, batch_22);

    Convolution conv_9_3(conv_23 , 64, 5, 1, 0, 1);
    output = conv_9_3.Convolution2D_GeMM(output, height, width);
    // output = conv(output, height, width, 64, 5, conv_23, 1, 0);
    
    output_image = Softmax(output, height, width, 5);

    WriteImage(path2save, output_image, height, width);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = (t2 - t1) / 1000;
    printf("Total time = %f seconds\n", ms_double);
    return 0;
}