using Matrix3D = std::vector<std::vector<std::vector<float>>>;
/*
Функция считывает изображение в вектор char и переводит в 3D массив с 3 каналами RGB типов int.
[каналы, высота, ширина]
*/
void ReadImage(const char* path, Matrix3D& output, unsigned int& width, unsigned int& height);

/*
Функция переводит 3D масив в вектор типов char с 4 каналами RGBA и записывает в файл.
*/
void WriteImage(const char* path, const Matrix3D& image3d ,unsigned int& width, unsigned int& height);