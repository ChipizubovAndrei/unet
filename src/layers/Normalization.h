#pragma once

/*
Функция нормализации входного изображения
Аргументы:
- src - возвращаемое нормализованное изображение
- srcH - высота изображения
- srcW - ширина изображения
- srcC - количество каналов в изображениии
*/
void Normalization(float* src, const int& srcH, const int& srcW, const int& srcC);