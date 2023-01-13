#include <string.h>
#include <cmath>
#include "paramgetter/paramgetter.h"
#include "BatchNormalization.h"

void BatchNormalization::BatchNormalization2D(float* src, unsigned int& srcH, unsigned int& srcW)
{
    float eps = 0.001;
    for (int sy = 0; sy < (int)srcH; sy++)
    {
        for (int sx = 0; sx < (int)srcW; sx++)
        {
            for (int sc = 0; sc < m_srcC; sc++)
            {
                src[(sy*srcH + sx)*m_srcC + sc] = m_gamma[sc]*(src[(sy*srcH + sx)*m_srcC + sc] - m_running_mean[sc]) / pow(m_running_var[sc] + eps, 0.5) + m_beta[sc];
            }
        }
    }
}

BatchNormalization::BatchNormalization(char* path_p2, int srcC)
{
    m_srcC = srcC;

    m_beta = new float [m_srcC];
    m_gamma = new float [m_srcC];
    m_running_mean = new float [m_srcC];
    m_running_var = new float [m_srcC];

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

    ParameterGetter(path_beta, m_beta, srcC, 0, 0);
    ParameterGetter(path_gamma, m_gamma, srcC, 0, 0);
    ParameterGetter(path_mean, m_running_mean, srcC, 0, 0);
    ParameterGetter(path_var, m_running_var, srcC, 0, 0);
};

BatchNormalization::~BatchNormalization()
{
    // std::cout << "Destroy BatchNormalization Layer" << std::endl;
    delete [] m_beta;
    delete [] m_gamma;
    delete [] m_running_mean;
    delete [] m_running_var;
};