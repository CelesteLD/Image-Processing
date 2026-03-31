# Image Processing — Computación en la Nube

**Máster en Ingeniería Informática — Universidad de La Laguna**  
**Autora:** Celeste Luis Díaz

---

## Descripción

Este repositorio contiene las prácticas de la asignatura **Computación en la Nube**, centradas en el procesamiento de imágenes PNG mediante un filtro de convolución de realce (*Sharpen*). A lo largo de las prácticas se desarrollan e implementan distintos modelos de paralelización, comparando su rendimiento frente a una versión secuencial de referencia.

El filtro aplicado en todas las versiones es el siguiente:

```
 0  -1   0
-1   5  -1
 0  -1   0
```

---

## Estructura del repositorio

```
.
├── images/                     # Imágenes de entrada y salida
│   ├── small.png               # Imagen pequeña
│   ├── medium.png              # Imagen mediana
│   ├── large.png               # Imagen grande
│   └── outputs/                # Resultados generados por cada versión
│       ├── secuencial/
│       ├── lab02/
│       └── lab03/
│
├── secuencial/                 # Versión secuencial de referencia
│   └── main.cpp
│
├── lab02/                      # Práctica 2 — Paralelización con OpenMP
│   ├── main.cpp
│   ├── image_omp
│   └── README.md
│
└── lab03/                      # Práctica 3 — Paralelización con MPI y CUDA
    ├── main_mpi.cpp
    ├── main_cuda.cu
    ├── image_mpi
    ├── image_cuda
    └── README.md
```

---

## Prácticas

### Secuencial — Versión de referencia

Implementación base del filtro de convolución sin paralelismo. Sirve como punto de comparación para medir el speedup de las versiones paralelas.

```bash
cd secuencial/
g++ main.cpp -lpng -o image_seq
./image_seq ../images/small.png ../images/outputs/secuencial/out_small.png
```

---

### Lab 02 — OpenMP

Paralelización del filtro sobre CPU mediante memoria compartida con OpenMP. Explora el impacto del número de hilos en el rendimiento y analiza las limitaciones del modelo *memory-bound*.

```bash
cd lab02/
g++ -fopenmp main.cpp -lpng -o image_omp
OMP_NUM_THREADS=N ./image_omp ../images/small.png ../images/outputs/lab02/out_small.png
```

> Consulta el [README de Lab 02](lab02/README.md) para más detalles.

---

### Lab 03 — MPI y CUDA

Incorpora dos nuevos modelos de paralelización:

- **MPI:** distribución del trabajo entre múltiples procesos independientes mediante paso de mensajes.
- **CUDA:** ejecución masivamente paralela en GPU, asignando un hilo por píxel de salida.

```bash
cd lab03/

# MPI — con N procesos
mpic++ main_mpi.cpp -lpng -o image_mpi
mpirun -np N ./image_mpi ../images/small.png ../images/outputs/lab03/out_small_mpi.png

# CUDA
nvcc main_cuda.cu -lpng -o image_cuda
./image_cuda ../images/small.png ../images/outputs/lab03/out_small_cu.png
```

> Consulta el [README de Lab 03](lab03/README.md) para más detalles.

---

## Comparativa de rendimiento

Speedup del tiempo de cómputo respecto a la versión secuencial para la imagen `large.png`:

| Versión          | T. cómputo (s) | Speedup |
|------------------|----------------|---------|
| Secuencial       | 2.875          | 1×      |
| OpenMP (1 hilo)  | 0.147          | ~19×    |
| MPI (4 proc.)    | 0.087          | ~33×    |
| CUDA             | 0.002          | ~1369×  |

> CUDA ofrece el mayor speedup en cómputo puro, aunque las transferencias CPU–GPU limitan su rendimiento global. MPI resulta más competitivo en tiempo total de ejecución para imágenes grandes.

---

## Dependencias

| Herramienta   | Uso                        |
|---------------|----------------------------|
| `g++`         | Compilación C++ (seq, OMP) |
| `libpng`      | Carga y guardado de PNG    |
| `OpenMP`      | Paralelismo en CPU         |
| `OpenMPI`     | Paralelismo distribuido    |
| `CUDA Toolkit`| Aceleración en GPU         |

Instalación en Ubuntu/Debian:

```bash
sudo apt install libpng-dev openmpi-bin openmpi-common libopenmpi-dev
```
