#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

/*
 * Filtro Sharpen (realza bordes y detalles)
 * Sustituye al Gaussiano
 */
Matrix getSharpen() {
    Matrix kernel = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    return kernel;
}

/*
 * Carga la imagen PNG en una estructura de 3 canales (RGB)
 */
Image loadImage(const char *filename)
{
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    for (int h = 0; h < image.get_height(); h++) {
        for (int w = 0; w < image.get_width(); w++) {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }

    return imageMatrix;
}

/*
 * Guarda la imagen procesada en disco
 */
void saveImage(Image &image, string filename)
{
    assert(image.size() == 3);

    int height = image[0].size();
    int width = image[0][0].size();

    png::image<png::rgb_pixel> imageFile(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageFile[y][x].red   = image[0][y][x];
            imageFile[y][x].green = image[1][y][x];
            imageFile[y][x].blue  = image[2][y][x];
        }
    }

    imageFile.write(filename);
}

Image applyFilter(Image &image, Matrix &filter)
{
    assert(image.size() == 3 && filter.size() != 0);

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    int newImageHeight = height - filterHeight + 1;
    int newImageWidth  = width  - filterWidth  + 1;

    /*
     * Inicialización explícita a 0.0
     * Esto evita valores basura y hace el comportamiento más claro y eficiente.
     */
    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth, 0.0)));

    /*
     * Paralelización:
     * - collapse(3): paraleliza d, i, j (RGB + filas + columnas)
     * - Cada hilo trabaja en píxeles independientes, por tanto no hay dependencias
     * - private asegura variables locales por hilo
     */
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < 3; d++) {
        for (int i = 0; i < newImageHeight; i++) {
            for (int j = 0; j < newImageWidth; j++) {

                /*
                 * Uso de variable local 'sum':
                 * evita acumulaciones directas en memoria compartida
                 * Esto permite código más seguro y eficiente (mejor uso de caché)
                 */
                double sum = 0.0;

                for (int h = i; h < i + filterHeight; h++) {
                    for (int w = j; w < j + filterWidth; w++) {
                        sum += filter[h - i][w - j] * image[d][h][w];
                    }
                }

                newImage[d][i][j] = sum;
            }
        }
    }

    return newImage;
}

/*
 * Permite aplicar el filtro varias veces
 */
Image applyFilter(Image &image, Matrix &filter, int times)
{
    Image newImage = image;
    for (int i = 0; i < times; i++) {
        newImage = applyFilter(newImage, filter);
    }
    return newImage;
}

int main(int argc, char *argv[])
{
    /*
     * Configuración del número de hilos
     * Esto es para el análisis de rendimiento
     */
    omp_set_num_threads(4);

    auto t1 = std::chrono::high_resolution_clock::now();

    Matrix filter = getSharpen();

    cout << "Loading image..." << endl;
    Image image = loadImage(argv[1]);

    cout << "Applying filter..." << endl;

    auto t1_1 = std::chrono::high_resolution_clock::now();

    Image newImage = applyFilter(image, filter);

    auto t2_1 = std::chrono::high_resolution_clock::now();

    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2_1 - t1_1).count();
    cout << "Tiempo de computo: " << (float)(duration_1 / 1000.0) << " sec" << endl;

    cout << "Saving image..." << endl;

    stringstream ss;
    ss << argv[2];
    string ficheroGuardar = ss.str();

    saveImage(newImage, ficheroGuardar);

    cout << "Done!" << endl;

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    cout << "Tiempo total de ejecucion: " << (float)(duration / 1000.0) << " sec" << endl;

    return 0;
}