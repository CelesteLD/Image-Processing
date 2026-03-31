/*
 * Procesamiento de imágenes con CUDA
 * Filtro Sharpen paralelizado con CUDA
 *
 * Compilar: nvcc main_cuda.cu -lpng -o image_cuda
 * Ejecutar: ./image_cuda <input.png> <output.png>
 */

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include <string>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

/* =========================================================
 * Filtro Sharpen (almacenado en memoria constante de GPU)
 * ========================================================= */
#define FILTER_H 3
#define FILTER_W 3

__constant__ float d_filter[FILTER_H * FILTER_W];

/* =========================================================
 * Kernel CUDA: convolución 2D por canal
 *
 * Estrategia:
 *  - Un hilo por píxel de salida (i, j)
 *  - Se lanzan 3 bloques en la dimensión z (uno por canal RGB)
 *  - Se usa memoria compartida (shared memory) para cargar
 *    un tile de la imagen de entrada, reduciendo accesos a
 *    memoria global (que es lenta).
 *
 * TILE_W x TILE_H: tamaño del tile de salida por bloque.
 * El tile de entrada es (TILE_W + FILTER_W - 1) x (TILE_H + FILTER_H - 1).
 * ========================================================= */
#define TILE_W 16
#define TILE_H 16

__global__ void applyFilterKernel(
    const float* __restrict__ input,   // imagen de entrada, todos los canales
    float*       output,               // imagen de salida, todos los canales
    int inHeight, int inWidth,
    int outHeight, int outWidth)
{
    int channel = blockIdx.z;   // canal RGB (0=R, 1=G, 2=B)

    // Índice de salida (píxel que computa este hilo)
    int outRow = blockIdx.y * TILE_H + threadIdx.y;
    int outCol = blockIdx.x * TILE_W + threadIdx.x;

    // Dimensiones del tile de entrada compartido
    const int sharedH = TILE_H + FILTER_H - 1;
    const int sharedW = TILE_W + FILTER_W - 1;

    __shared__ float sharedTile[sharedH * sharedW];

    // Cada hilo carga uno o más píxeles en shared memory
    // Coordenada de inicio en la imagen de entrada para este bloque
    int inRowBase = blockIdx.y * TILE_H;
    int inColBase = blockIdx.x * TILE_W;

    // Llenado cooperativo del tile compartido
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int totalShared = sharedH * sharedW;

    for (int idx = threadId; idx < totalShared; idx += blockDim.x * blockDim.y) {
        int sr = idx / sharedW;
        int sc = idx % sharedW;
        int inR = inRowBase + sr;
        int inC = inColBase + sc;

        if (inR < inHeight && inC < inWidth) {
            sharedTile[idx] = input[channel * inHeight * inWidth + inR * inWidth + inC];
        } else {
            sharedTile[idx] = 0.0f;
        }
    }

    __syncthreads(); // esperar a que todos carguen el tile

    // Calcular convolución para el píxel asignado a este hilo
    if (outRow < outHeight && outCol < outWidth) {
        float sum = 0.0f;
        for (int fh = 0; fh < FILTER_H; fh++) {
            for (int fw = 0; fw < FILTER_W; fw++) {
                sum += d_filter[fh * FILTER_W + fw] *
                       sharedTile[(threadIdx.y + fh) * sharedW + (threadIdx.x + fw)];
            }
        }
        output[channel * outHeight * outWidth + outRow * outWidth + outCol] = sum;
    }
}

/* =========================================================
 * Macro de comprobación de errores CUDA
 * ========================================================= */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            cerr << "CUDA error in " << __FILE__ << " line " << __LINE__   \
                 << ": " << cudaGetErrorString(err) << endl;                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

/* =========================================================
 * Utilidades de E/S de imagen
 * ========================================================= */
Image loadImage(const char *filename) {
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    for (int h = 0; h < (int)image.get_height(); h++) {
        for (int w = 0; w < (int)image.get_width(); w++) {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }
    return imageMatrix;
}

void saveImage(Image &image, const string &filename) {
    assert(image.size() == 3);
    int height = image[0].size();
    int width  = image[0][0].size();

    png::image<png::rgb_pixel> imageFile(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageFile[y][x].red   = (png::byte)max(0.0, min(255.0, image[0][y][x]));
            imageFile[y][x].green = (png::byte)max(0.0, min(255.0, image[1][y][x]));
            imageFile[y][x].blue  = (png::byte)max(0.0, min(255.0, image[2][y][x]));
        }
    }
    imageFile.write(filename);
}

/* =========================================================
 * main
 * ========================================================= */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Uso: " << argv[0] << " <input.png> <output.png>" << endl;
        return 1;
    }

    auto t_total_start = chrono::high_resolution_clock::now();

    /* ----- Filtro Sharpen ----- */
    float h_filter[FILTER_H * FILTER_W] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    /* Copiar filtro a memoria constante GPU */
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter)));

    /* ----- Cargar imagen ----- */
    cout << "Cargando imagen..." << endl;
    Image image = loadImage(argv[1]);

    int inHeight = image[0].size();
    int inWidth  = image[0][0].size();
    int outHeight = inHeight - FILTER_H + 1;
    int outWidth  = inWidth  - FILTER_W + 1;

    int inPixels  = 3 * inHeight  * inWidth;
    int outPixels = 3 * outHeight * outWidth;

    /* ----- Preparar datos de entrada planos (float) ----- */
    vector<float> h_input(inPixels);
    for (int d = 0; d < 3; d++)
        for (int r = 0; r < inHeight; r++)
            for (int c = 0; c < inWidth; c++)
                h_input[d * inHeight * inWidth + r * inWidth + c] = (float)image[d][r][c];

    /* ----- Reservar memoria en GPU ----- */
    float *d_input  = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input,  inPixels  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, outPixels * sizeof(float)));

    /* ----- Transferir imagen a GPU (Host → Device) ----- */
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), inPixels * sizeof(float), cudaMemcpyHostToDevice));

    /* ----- Configurar lanzamiento del kernel ----- */
    // Dimensiones de bloque: TILE_W x TILE_H hilos
    dim3 blockDim(TILE_W, TILE_H, 1);

    // Dimensiones de grid: celdas necesarias para cubrir la imagen de salida + 3 canales
    dim3 gridDim(
        (outWidth  + TILE_W - 1) / TILE_W,
        (outHeight + TILE_H - 1) / TILE_H,
        3  // un plano z por canal RGB
    );

    cout << "Aplicando filtro en GPU..." << endl;
    cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << endl;
    cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << endl;

    /* ----- Lanzar kernel y medir tiempo de cómputo ----- */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    applyFilterKernel<<<gridDim, blockDim>>>(d_input, d_output,
                                             inHeight, inWidth,
                                             outHeight, outWidth);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    CUDA_CHECK(cudaGetLastError()); // verificar errores del kernel

    float ms_compute = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_compute, ev_start, ev_stop));
    cout << "Tiempo de computo (kernel): " << ms_compute / 1000.0f << " sec" << endl;

    /* ----- Transferir resultado a CPU (Device → Host) ----- */
    vector<float> h_output(outPixels);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, outPixels * sizeof(float), cudaMemcpyDeviceToHost));

    /* ----- Reconstruir imagen de salida ----- */
    Image newImage(3, Matrix(outHeight, Array(outWidth, 0.0)));
    for (int d = 0; d < 3; d++)
        for (int r = 0; r < outHeight; r++)
            for (int c = 0; c < outWidth; c++)
                newImage[d][r][c] = (double)h_output[d * outHeight * outWidth + r * outWidth + c];

    /* ----- Liberar memoria GPU ----- */
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    cout << "Guardando imagen..." << endl;
    saveImage(newImage, string(argv[2]));
    cout << "Listo!" << endl;

    auto t_total_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(t_total_end - t_total_start).count();
    cout << "Tiempo total de ejecucion: " << duration / 1000.0f << " sec" << endl;

    return 0;
}
