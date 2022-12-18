#include <vector>
#include <string.h>
#include <cmath>
#include "paramgetter/paramgetter.h"
#include "BatchNormalization.h"

void BatchNormalization::BatchNormalization2D(Matrix3D& prev_layer_out, unsigned int& in_height, unsigned int& in_width)
{
    float eps = 0.001;
    for (int y = 0; y < (int)in_height; y++)
    {
        for (int x = 0; x < (int)in_width; x++)
        {
            for (int ch = 0; ch < m_in_channels; ch++)
            {
                prev_layer_out[y][x][ch] = m_gamma[ch]*(prev_layer_out[y][x][ch] - m_running_mean[ch]) / pow(m_running_var[ch] + eps, 0.5) + m_beta[ch];
                // prev_layer_out[y][x][ch] = m_gamma[ch]*prev_layer_out[y][x][ch] + m_beta[ch];
            }
        }
    }
}

BatchNormalization::BatchNormalization(char* path_p2, int in_channels)
{
    m_in_channels = in_channels;

    m_beta = new float [m_in_channels];
    m_gamma = new float [m_in_channels];
    m_running_mean = new float [m_in_channels];
    m_running_var = new float [m_in_channels];

    char path_beta[strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_beta_p3) + 1] = "";
    char path_gamma [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_gamma_p3) + 1] = "";
    char path_mean [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_mean_p3) + 1] = "";
    char path_var [strlen(m_path_p1) + strlen(path_p2) + strlen(m_path_var_p3) + 1] = "";

    strcat(path_beta, m_path_p1);
    strcat(path_beta, path_p2);
    strcat(path_beta, m_path_beta_p3);

    strcat(path_gamma, m_path_p1);
    strcat(path_gamma, path_p2);
    strcat(path_gamma, m_path_gamma_p3);

    strcat(path_mean, m_path_p1);
    strcat(path_mean, path_p2);
    strcat(path_mean, m_path_mean_p3);

    strcat(path_var, m_path_p1);
    strcat(path_var, path_p2);
    strcat(path_var, m_path_var_p3);

    ParameterGetter(path_beta, m_beta, in_channels, 0, 0);
    ParameterGetter(path_gamma, m_gamma, in_channels, 0, 0);
    ParameterGetter(path_mean, m_running_mean, in_channels, 0, 0);
    ParameterGetter(path_var, m_running_var, in_channels, 0, 0);
};

BatchNormalization::~BatchNormalization()
{
    // std::cout << "Destroy BatchNormalization Layer" << std::endl;
    delete [] m_beta;
    delete [] m_gamma;
    delete [] m_running_mean;
    delete [] m_running_var;
};