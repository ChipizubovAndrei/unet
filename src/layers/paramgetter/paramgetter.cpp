#include "paramgetter.h"
#include "H5Cpp.h"

using namespace H5;
using namespace std;

void ParameterGetter(const char* path, float* data_out, const int out_channels, const int in_channels, const int kernel_size)
{
    const H5std_string FILE_NAME( "../parameters/unet(leakyrelu).hdf5" );
    const H5std_string DATASET_NAME( path );

    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    DataSet dataset = file.openDataSet( DATASET_NAME );

    DataSpace dataspace = dataset.getSpace();

    hsize_t offset[4];   // hyperslab offset in the file
    hsize_t count[4];    // size of the hyperslab in the file

    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    offset[3] = 0;

    count[0] = kernel_size;
    count[1] = kernel_size;
    count[2] = in_channels;
    count[3] = out_channels;

    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
    /*
    * Define the memory dataspace.
    */
    hsize_t dimsm[1];              /* memory space dimensions */
    dimsm[0] = kernel_size*kernel_size*in_channels*out_channels;
    DataSpace memspace(1, dimsm );
    /*
    * Define memory hyperslab.
    */
    hsize_t offset_out[1];   // hyperslab offset in memory
    hsize_t count_out[1];    // size of the hyperslab in memory

    offset_out[0] = 0;

    count_out[0] = kernel_size*kernel_size*in_channels*out_channels;
    
    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out);

    dataset.read( data_out, PredType::NATIVE_FLOAT);
}