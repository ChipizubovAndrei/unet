#pragma once

/*
Функция активации ReLU
Аргументы:
    - src - возвращаемое изображение
    - srcH - высота изображения
    - srcW - ширина изображения
    - srcC - количество слоев
*/
void ReLU(float* src, const int& srcH, const int& srcW, const int& srcC);

/*
Функция активации LeakyReLU
Аргументы:
    - src - возвращаемое изображение
    - srcH - высота изображения
    - srcW - ширина изображения
    - srcC - количество слоев
    - negative_slope - параметр наклона прямой отрицательной части
*/
void LeakyReLU(float* src, const int& srcH, const int& srcW, const int& srcC, float negative_slope);

/*
Функция активации ELU
Аргументы:
    - src - возвращаемое изображение
    - srcH - высота изображения
    - srcW - ширина изображения
    - srcC - количество слоев
    - alpha - коэффициент крутизны кривой
*/
void ELU(float* src, const int& srcH, const int& srcW, const int& srcC, float alpha);

/*
Функция активации Softmax
Аргументы:
    - src - возвращаемое изображение
    - srcH - высота изображения
    - srcW - ширина изображения
    - srcC - количество слоев
*/
float* Softmax(float* src, const int& srcH, const int& srcW, const int& srcC);