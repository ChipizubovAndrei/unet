/*
Функция считывает изображение в вектор char и переводит в 1D массив с 3 каналами RGB типов int.
[каналы, высота, ширина]
Аргументы:
- path - путь к изображению
- output - возвращаемый массив
- srcH - возвращаемое значание высоты изображения
- srcW - возврщаемое значение ширины изображения
*/
void ReadImage(const char* path, float*& output, unsigned int& srcH, unsigned int& srcW);

/*
Функция переводит 3D масив в вектор типов char с 4 каналами RGBA и записывает в файл.
Аргументы:
- path - путь к изображению
- src - входное изображение
- srcH - высота изображения
- srcW - ширина изображения
*/
void WriteImage(const char* path, float* src ,unsigned int& srcH, unsigned int& srcW);