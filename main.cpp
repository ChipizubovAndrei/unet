#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <ctime>>
#include "src/datahandler/datahandler.h"
#include "src/layers/Convolutional.h"
#include "src/layers/MaxPooling.h"
#include "src/layers/UpSampling.h"
#include "src/layers/BatchNormalization.h"
#include "src/layers/Concatenation.h"
#include "src/activation/Activations.h"
#include "src/layers/Normalization.h"

#include "src/layers/paramgetter/paramgetter.h"

using namespace std;

int main()
{
    time_t start, end;
    time(&start);
    typedef std::vector<std::vector<std::vector<float>>> Matrix3D;
    unsigned int height, width;
    height = 256;
    width = 256;
    // float alpha = 0.08;
    
    Matrix3D original_image;
    Matrix3D output, output_blok_1, output_blok_2, output_blok_3, output_blok_4, output_image;
    const char* path2image = "/home/galahad/Documents/5_course/1_semester/SRW/data/Abies Sibirica Dataset/Abies Sibirica Dataset/fragments/validation/x/103.png";
    const char* path2save = "/home/galahad/Documents/5_course/1_semester/SRW/cpp_unet/mask_103.png";

    // Считавание изображения
    ReadImage(path2image, original_image, width, height);
    Normalization(original_image, height, width, 3);

    cout << "Первый блок (1)" << endl;
    // Первый блок (1)
    Convolution conv_1_1("conv2d/conv2d", 3, 64, 3, 1, 1);
    output = conv_1_1.Convolution2D(original_image, height, width);
    ReLU(output, height, width, 64);
    BatchNormalization batch_1_1("batch_normalization/batch_normalization", 64);
    batch_1_1.BatchNormalization2D(output, height, width);

    cout << "Первый блок (2)" << endl;
    // Первый блок (2)
    Convolution conv_1_2("conv2d_1/conv2d_1", 64, 64, 3, 1, 1);
    output_blok_1 = conv_1_2.Convolution2D(output, height, width);
    ReLU(output_blok_1, height, width, 64);
    BatchNormalization batch_1_2("batch_normalization_1/batch_normalization_1", 64);
    batch_1_2.BatchNormalization2D(output_blok_1, height, width);

    MaxPool pool_1(64, 2, 0, 2);
    output = pool_1.MaxPool2D(output_blok_1, height, width);


    cout << "Второй блок (1)" << endl;
    // Второй блок (1)
    Convolution conv_2_1("conv2d_2/conv2d_2", 64, 128, 3, 1, 1);
    output = conv_2_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 128);
    BatchNormalization batch_2_1("batch_normalization_2/batch_normalization_2", 128);
    batch_2_1.BatchNormalization2D(output, height, width);

    cout << "Второй блок (2)" << endl;
    // Второй блок (2)
    Convolution conv_2_2("conv2d_3/conv2d_3", 128, 128, 3, 1, 1);
    output_blok_2 = conv_2_2.Convolution2D(output, height, width);
    ReLU(output_blok_2, height, width, 128);
    BatchNormalization batch_2_2("batch_normalization_3/batch_normalization_3", 128);
    batch_2_2.BatchNormalization2D(output_blok_2, height, width);

    MaxPool pool_2(128, 2, 0, 2);
    output = pool_2.MaxPool2D(output_blok_2, height, width);


    cout << "Третий блок (1)" << endl;
    // Третий блок (1)
    Convolution conv_3_1("conv2d_4/conv2d_4", 128, 256, 3, 1, 1);
    output = conv_3_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 256);
    BatchNormalization batch_3_1("batch_normalization_4/batch_normalization_4", 256);
    batch_3_1.BatchNormalization2D(output, height, width);

    cout << "Третий блок (2)" << endl;
    // Третий блок (2)
    Convolution conv_3_2("conv2d_5/conv2d_5", 256, 256, 3, 1, 1);
    output_blok_3 = conv_3_2.Convolution2D(output, height, width);
    ReLU(output_blok_3, height, width, 256);
    BatchNormalization batch_3_2("batch_normalization_5/batch_normalization_5", 256);
    batch_3_2.BatchNormalization2D(output_blok_3, height, width);

    MaxPool pool_3(256, 2, 0, 2);
    output = pool_3.MaxPool2D(output_blok_3, height, width);


    cout << "Четвертый блок (1)" << endl;
    // Четвертый блок (1)
    Convolution conv_4_1("conv2d_6/conv2d_6", 256, 512, 3, 1, 1);
    output = conv_4_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 512);
    BatchNormalization batch_4_1("batch_normalization_6/batch_normalization_6", 512);
    batch_4_1.BatchNormalization2D(output, height, width);

    cout << "Четвертый блок (2)" << endl;
    // Четвертый блок (2)
    Convolution conv_4_2("conv2d_7/conv2d_7", 512, 512, 3, 1, 1);
    output_blok_4 = conv_4_2.Convolution2D(output, height, width);
    ReLU(output_blok_4, height, width, 512);
    BatchNormalization batch_4_2("batch_normalization_7/batch_normalization_7", 512);
    batch_4_2.BatchNormalization2D(output_blok_4, height, width);

    MaxPool pool_4(512, 2, 0, 2);
    output = pool_4.MaxPool2D(output_blok_4, height, width);


    cout << "Пятый блок (1)" << endl;
    // Пятый блок (1)
    Convolution conv_5_1("conv2d_8/conv2d_8", 512, 1024, 3, 1, 1);
    output = conv_5_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 1024);
    BatchNormalization batch_5_1("batch_normalization_8/batch_normalization_8", 1024);
    batch_5_1.BatchNormalization2D(output, height, width);

    cout << "Пятый блок (2)" << endl;
    // Пятый блок (2)
    Convolution conv_5_2("conv2d_9/conv2d_9", 1024, 1024, 3, 1, 1);
    output = conv_5_2.Convolution2D(output, height, width);
    ReLU(output, height, width, 1024);
    BatchNormalization batch_5_2("batch_normalization_9/batch_normalization_9", 1024);
    batch_5_2.BatchNormalization2D(output, height, width);

    cout << "Пятый блок (3)" << endl;
    // Пятый блок (3)
    UpSample up_5 = UpSample(1024, 2);
    output = up_5.UpSample2D(output, height, width);

    Convolution conv_5_3("conv2d_10/conv2d_10", 1024, 512, 3, 1, 1);
    output = conv_5_3.Convolution2D(output, height, width);
    ReLU(output, height, width, 512);
    BatchNormalization batch_5_3("batch_normalization_10/batch_normalization_10", 512);
    batch_5_3.BatchNormalization2D(output, height, width);

    Concatenation merge_5 = Concatenation(512, 1024);
    output = merge_5.Concatenation2D(output, output_blok_4, height, width);


    cout << "Шестой блок (1)" << endl;
    // Шестой блок (1)
    Convolution conv_6_1("conv2d_11/conv2d_11", 1024, 512, 3, 1, 1);
    output = conv_6_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 512);
    BatchNormalization batch_6_1("batch_normalization_11/batch_normalization_11", 512);
    batch_6_1.BatchNormalization2D(output, height, width);

    cout << "Шестой блок (2)" << endl;
    // Шестой блок (2)
    Convolution conv_6_2("conv2d_12/conv2d_12", 512, 512, 3, 1, 1);
    output = conv_6_2.Convolution2D(output, height, width);
    ReLU(output, height, width, 512);
    BatchNormalization batch_6_2("batch_normalization_12/batch_normalization_12", 512);
    batch_6_2.BatchNormalization2D(output, height, width);

    cout << "Шестой блок (3)" << endl;
    // Шестой блок (3)
    UpSample up_6 = UpSample(512, 2);
    output = up_6.UpSample2D(output, height, width);

    Convolution conv_6_3("conv2d_13/conv2d_13", 512, 256, 3, 1, 1);
    output = conv_6_3.Convolution2D(output, height, width);
    ReLU(output, height, width, 256);
    BatchNormalization batch_6_3("batch_normalization_13/batch_normalization_13", 256);
    batch_6_3.BatchNormalization2D(output, height, width);

    Concatenation merge_6 = Concatenation(256, 512);
    output = merge_6.Concatenation2D(output, output_blok_3, height, width);


    cout << "Седьмой блок (1)" << endl;
    // Седьмой блок (1)
    Convolution conv_7_1("conv2d_14/conv2d_14", 512, 256, 3, 1, 1);
    output = conv_7_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 256);
    BatchNormalization batch_7_1("batch_normalization_14/batch_normalization_14", 256);
    batch_7_1.BatchNormalization2D(output, height, width);

    cout << "Седьмой блок (2)" << endl;
    // Седьмой блок (2)
    Convolution conv_7_2("conv2d_15/conv2d_15", 256, 256, 3, 1, 1);
    output = conv_7_2.Convolution2D(output, height, width);
    ReLU(output, height, width, 256);
    BatchNormalization batch_7_2("batch_normalization_15/batch_normalization_15", 256);
    batch_7_2.BatchNormalization2D(output, height, width);

    cout << "Седьмой блок (3)" << endl;
    // Седьмой блок (3)
    UpSample up_7 = UpSample(256, 2);
    output = up_7.UpSample2D(output, height, width);

    Convolution conv_7_3("conv2d_16/conv2d_16", 256, 128, 3, 1, 1);
    output = conv_7_3.Convolution2D(output, height, width);
    ReLU(output, height, width, 128);
    BatchNormalization batch_7_3("batch_normalization_16/batch_normalization_16", 128);
    batch_7_3.BatchNormalization2D(output, height, width);

    Concatenation merge_7 = Concatenation(128, 256);
    output = merge_7.Concatenation2D(output, output_blok_2, height, width);


    cout << "Восьмой блок (1)" << endl;
    // Восьмой блок (1)
    Convolution conv_8_1("conv2d_17/conv2d_17", 256, 128, 3, 1, 1);
    output = conv_8_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 128);
    BatchNormalization batch_8_1("batch_normalization_17/batch_normalization_17", 128);
    batch_8_1.BatchNormalization2D(output, height, width);

    cout << "Восьмой блок (2)" << endl;
    // Восьмой блок (2)
    Convolution conv_8_2("conv2d_18/conv2d_18", 128, 128, 3, 1, 1);
    output = conv_8_2.Convolution2D(output, height, width);
    ReLU(output, height, width, 128);
    BatchNormalization batch_8_2("batch_normalization_18/batch_normalization_18", 128);
    batch_8_2.BatchNormalization2D(output, height, width);

    cout << "Восьмой блок (3)" << endl;
    // Восьмой блок (3)
    UpSample up_8 = UpSample(128, 2);
    output = up_8.UpSample2D(output, height, width);

    Convolution conv_8_3("conv2d_19/conv2d_19", 128, 64, 3, 1, 1);
    output = conv_8_3.Convolution2D(output, height, width);
    ReLU(output, height, width, 64);
    BatchNormalization batch_8_3("batch_normalization_19/batch_normalization_19", 64);
    batch_8_3.BatchNormalization2D(output, height, width);

    Concatenation merge_8 = Concatenation(64, 128);
    output = merge_8.Concatenation2D(output, output_blok_1, height, width);

    
    cout << "Девятый блок (1)" << endl;
    // Девятый блок (1)
    Convolution conv_9_1("conv2d_20/conv2d_20", 128, 64, 3, 1, 1);
    output = conv_9_1.Convolution2D(output, height, width);
    ReLU(output, height, width, 64);
    BatchNormalization batch_9_1("batch_normalization_20/batch_normalization_20", 64);
    batch_9_1.BatchNormalization2D(output, height, width);

    cout << "Девятый блок (2)" << endl;
    // Девятый блок (2)
    Convolution conv_9_2("conv2d_21/conv2d_21", 64, 64, 3, 1, 1);
    output = conv_9_2.Convolution2D(output, height, width);
    ReLU(output, height, width, 64);
    BatchNormalization batch_9_2("batch_normalization_21/batch_normalization_21", 64);
    batch_9_2.BatchNormalization2D(output, height, width);

    cout << "Девятый блок (3)" << endl;
    // Девятый блок (3)
    Convolution conv_9_3("conv2d_22/conv2d_22", 64, 5, 1, 0, 1);
    output = conv_9_3.Convolution2D(output, height, width);

    output_image = Softmax(output, height, width, 5);

    WriteImage(path2save, output_image, width, height);

    time(&end);

    double seconds = difftime(end, start);
    printf("The time: %f seconds\n", seconds);

    return 0;
}