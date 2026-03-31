/*
 * Procesamiento de imágenes con MPI
 * Filtro Sharpen paralelizado con MPI
 *
 * Compilar: mpic++ main_mpi.cpp -lpng -o image_mpi
 * Ejecutar: mpirun -np <N> ./image_mpi <input.png> <output.png>
 *
 * Layout de buffers planos: [fila][canal][columna]
 * Es decir, el índice es: row * 3 * width + d * width + col
 * Este layout se usa de forma consistente en send, recv y gather
 * para evitar mezclas de canales entre procesos.
 */

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include <string>
#include <chrono>
#include <mpi.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

/* Filtro Sharpen */
Matrix getSharpen() {
    Matrix kernel = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    return kernel;
}

/* Carga la imagen PNG en una estructura de 3 canales (RGB) */
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

/* Guarda la imagen procesada en disco */
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

/*
 * Convierte Image (3 canales separados) a buffer plano con layout [fila][canal][col]
 * Indice: row * 3 * width + d * width + col
 */
vector<double> imageToBuffer(Image &image, int rows, int width) {
    vector<double> buf(rows * 3 * width);
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < 3; d++)
            for (int c = 0; c < width; c++)
                buf[r * 3 * width + d * width + c] = image[d][r][c];
    return buf;
}

/*
 * Convierte buffer plano [fila][canal][col] a Image (3 canales separados)
 */
Image bufferToImage(vector<double> &buf, int rows, int width) {
    Image image(3, Matrix(rows, Array(width, 0.0)));
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < 3; d++)
            for (int c = 0; c < width; c++)
                image[d][r][c] = buf[r * 3 * width + d * width + c];
    return image;
}

/*
 * Aplica el filtro sobre la imagen local completa.
 */
Image applyFilter(Image &image, Matrix &filter) {
    int filterHeight = filter.size();
    int filterWidth  = filter[0].size();
    int inRows       = image[0].size();
    int inWidth      = image[0][0].size();
    int outRows      = inRows  - filterHeight + 1;
    int outWidth     = inWidth - filterWidth  + 1;

    if (outRows <= 0 || outWidth <= 0)
        return Image(3, Matrix(0, Array(0)));

    Image result(3, Matrix(outRows, Array(outWidth, 0.0)));
    for (int d = 0; d < 3; d++)
        for (int i = 0; i < outRows; i++)
            for (int j = 0; j < outWidth; j++) {
                double sum = 0.0;
                for (int fh = 0; fh < filterHeight; fh++)
                    for (int fw = 0; fw < filterWidth; fw++)
                        sum += filter[fh][fw] * image[d][i + fh][j + fw];
                result[d][i][j] = sum;
            }
    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Uso: mpirun -np <N> " << argv[0] << " <input.png> <output.png>" << endl;
        MPI_Finalize();
        return 1;
    }

    double t_total_start = MPI_Wtime();

    Matrix filter = getSharpen();
    int filterHeight = (int)filter.size();
    int filterWidth  = (int)filter[0].size();

    int height = 0, width = 0;
    Image image;

    /* Solo el proceso raiz carga la imagen */
    if (rank == 0) {
        cout << "Cargando imagen..." << endl;
        image  = loadImage(argv[1]);
        height = image[0].size();
        width  = image[0][0].size();
    }

    /* Difundir dimensiones a todos los procesos */
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width,  1, MPI_INT, 0, MPI_COMM_WORLD);

    int outHeight = height - filterHeight + 1;
    int outWidth  = width  - filterWidth  + 1;

    /*
     * Distribucion por bandas de filas de SALIDA.
     * El proceso p produce outRows[p] filas de salida,
     * para lo cual necesita (outRows[p] + filterHeight - 1) filas de entrada.
     */
    int baseRows  = outHeight / size;
    int remainder = outHeight % size;

    vector<int> pOutRows(size), pOutStart(size);
    for (int p = 0; p < size; p++) {
        pOutRows[p]  = baseRows + (p < remainder ? 1 : 0);
        pOutStart[p] = (p == 0) ? 0 : pOutStart[p-1] + pOutRows[p-1];
    }

    vector<int> pInRows(size), pInStart(size);
    for (int p = 0; p < size; p++) {
        pInStart[p] = pOutStart[p];
        pInRows[p]  = pOutRows[p] + filterHeight - 1;
        if (pInStart[p] + pInRows[p] > height)
            pInRows[p] = height - pInStart[p];
    }

    /* Proceso raiz envia cada banda al proceso correspondiente */
    int myInRows = pInRows[rank];
    vector<double> localBuf(myInRows * 3 * width, 0.0);

    if (rank == 0) {
        /* Copiar la propia banda */
        for (int r = 0; r < pInRows[0]; r++)
            for (int d = 0; d < 3; d++)
                for (int c = 0; c < width; c++)
                    localBuf[r * 3 * width + d * width + c] =
                        image[d][pInStart[0] + r][c];

        /* Enviar al resto de procesos */
        for (int p = 1; p < size; p++) {
            int sendSize = pInRows[p] * 3 * width;
            vector<double> sendBuf(sendSize);
            for (int r = 0; r < pInRows[p]; r++)
                for (int d = 0; d < 3; d++)
                    for (int c = 0; c < width; c++)
                        sendBuf[r * 3 * width + d * width + c] =
                            image[d][pInStart[p] + r][c];
            MPI_Send(sendBuf.data(), sendSize, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(localBuf.data(), myInRows * 3 * width, MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Reconstruir subimagen local y aplicar filtro */
    Image localImage  = bufferToImage(localBuf, myInRows, width);

    double t_compute_start = MPI_Wtime();
    Image  localResult = applyFilter(localImage, filter);
    double t_compute_end   = MPI_Wtime();

    int myOutRows = (int)localResult[0].size();

    /* Empaquetar resultado con el mismo layout [fila][canal][col] */
    int localResultSize = myOutRows * 3 * outWidth;
    vector<double> resultBuf(localResultSize);
    for (int r = 0; r < myOutRows; r++)
        for (int d = 0; d < 3; d++)
            for (int c = 0; c < outWidth; c++)
                resultBuf[r * 3 * outWidth + d * outWidth + c] = localResult[d][r][c];

    /*
     * MPI_Gatherv reune todos los bloques en el proceso raiz.
     * Como el layout es [fila][canal][col] y los bloques son bandas contiguas
     * de filas, la concatenacion de bloques produce directamente el buffer
     * global correcto.
     */
    vector<int> recvCounts(size), displs(size);
    for (int p = 0; p < size; p++) {
        recvCounts[p] = pOutRows[p] * 3 * outWidth;
        displs[p]     = (p == 0) ? 0 : displs[p-1] + recvCounts[p-1];
    }

    vector<double> globalBuf;
    if (rank == 0)
        globalBuf.resize(outHeight * 3 * outWidth);

    MPI_Gatherv(resultBuf.data(), localResultSize, MPI_DOUBLE,
                globalBuf.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Proceso raiz reconstruye imagen y guarda */
    if (rank == 0) {
        Image newImage = bufferToImage(globalBuf, outHeight, outWidth);

        double t_total_end = MPI_Wtime();

        cout << "Guardando imagen..." << endl;
        saveImage(newImage, string(argv[2]));
        cout << "Listo!" << endl;
        cout << "Tiempo de computo (solo filtro): "
             << (t_compute_end - t_compute_start) << " sec" << endl;
        cout << "Tiempo total de ejecucion:       "
             << (t_total_end - t_total_start) << " sec" << endl;
    }

    MPI_Finalize();
    return 0;
}
