
# Lab 03 — Procesamiento de imágenes con MPI y CUDA

**Asignatura:** Computación en la Nube  
**Máster en Ingeniería Informática — Universidad de La Laguna**  
**Autora:** Celeste Luis Díaz

---

## Descripción

Esta práctica amplía el estudio del paralelismo iniciado en la Lab 02, incorporando dos nuevos modelos de ejecución aplicados al mismo filtro de convolución de realce (*Sharpen*):

- **MPI:** paralelización distribuida mediante paso de mensajes entre procesos independientes.
- **CUDA:** aceleración en GPU mediante ejecución masivamente paralela.

Se compara el rendimiento de las cuatro versiones disponibles (secuencial, OpenMP, MPI y CUDA) variando el número de procesos y el tamaño de las imágenes de entrada.

---

## Estructura del proyecto

```
.
├── images/
│   ├── small.png                   # Imagen de prueba pequeña
│   ├── medium.png                  # Imagen de prueba mediana
│   ├── large.png                   # Imagen de prueba grande
│   └── outputs/
│       └── lab03/                  # Salidas de las versiones MPI y CUDA
│           ├── out_small_mpi.png
│           ├── out_medium_mpi.png
│           ├── out_large_mpi.png
│           ├── out_small_cu.png
│           ├── out_medium_cu.png
│           └── out_large_cu.png
├── secuencial/
│   └── main.cpp                    # Implementación secuencial de referencia
├── lab02/
│   └── main.cpp                    # Implementación con OpenMP
└── lab03/
    ├── main_mpi.cpp                # Implementación distribuida con MPI
    ├── main_cuda.cu                # Implementación en GPU con CUDA
    ├── image_mpi                   # Binario compilado (MPI)
    ├── image_cuda                  # Binario compilado (CUDA)
    └── README.md
```

---

## Dependencias

- Compilador C++ (`g++`)
- MPI (`mpic++` / `mpirun`) — p. ej. OpenMPI
- CUDA Toolkit (`nvcc`)
- Librería `libpng`

Instalación en Ubuntu/Debian:

```bash
sudo apt install libpng-dev openmpi-bin openmpi-common libopenmpi-dev
```
---

## Compilación

### Versión MPI

```bash
cd lab03/
mpic++ main_mpi.cpp -lpng -o image_mpi
```

### Versión CUDA

```bash
cd lab03/
nvcc main_cuda.cu -lpng -o image_cuda
```

---

## Ejecución

### Versión MPI

```bash
cd lab03/
# Con N procesos
mpirun -np N ./image_mpi ../images/small.png ../images/outputs/lab03/out_small_mpi.png
```

### Versión CUDA

```bash
cd lab03/
./image_cuda ../images/small.png ../images/outputs/lab03/out_small_cu.png
```

---

## Resultados destacados

| Versión          | T. cómputo — large (s) | Speedup vs. secuencial |
|------------------|------------------------|------------------------|
| Secuencial       | 2.875                  | 1×                     |
| OpenMP (1 hilo)  | 0.147                  | ~19×                   |
| MPI (4 proc.)    | 0.087                  | ~33×                   |
| CUDA             | 0.002                  | ~1369×                 |

> Los tiempos totales incluyen operaciones de E/S y, en CUDA, transferencias de memoria entre CPU y GPU.

### Conclusiones principales

- **OpenMP** mejora notablemente el cómputo pero su escalabilidad está limitada por la naturaleza *memory-bound* del algoritmo.
- **MPI** escala mejor al aumentar el número de procesos, especialmente en imágenes grandes, aunque el overhead de comunicación limita la eficiencia en problemas pequeños.
- **CUDA** ofrece el mayor speedup en cómputo puro, pero las transferencias CPU–GPU dominan el tiempo total de ejecución.

---

## Repositorio

[https://github.com/CelesteLD/Image-Processing](https://github.com/CelesteLD/Image-Processing)