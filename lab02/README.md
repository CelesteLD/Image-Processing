# Lab 02 — Procesamiento de imágenes con OpenMP

**Asignatura:** Computación en la Nube  
**Máster en Ingeniería Informática — Universidad de La Laguna**  
**Autora:** Celeste Luis Díaz

---

## Descripción

Esta práctica aborda el procesamiento de imágenes PNG en formato RGB mediante la aplicación de un filtro de convolución de realce (*Sharpen*), definido por el siguiente kernel:

```
 0  -1   0
-1   5  -1
 0  -1   0
```

Se desarrollan dos versiones del algoritmo:

- **Secuencial:** implementación de referencia, ubicada en la carpeta `secuencial/`.
- **OpenMP:** paralelización del filtro sobre CPU mediante memoria compartida.

---

## Estructura del proyecto

```
.
├── images/
│   ├── small.png               # Imagen de prueba pequeña
│   ├── medium.png              # Imagen de prueba mediana
│   ├── large.png               # Imagen de prueba grande
│   └── outputs/
│       ├── secuencial/         # Salidas de la versión secuencial
│       │   ├── out_small.png
│       │   ├── out_medium.png
│       │   └── out_large.png
│       └── lab02/              # Salidas de la versión OpenMP
│           ├── out_small.png
│           ├── out_medium.png
│           └── out_large.png
├── secuencial/
│   └── main.cpp                # Implementación secuencial de referencia
└── lab02/
    ├── main.cpp                # Implementación paralela con OpenMP
    ├── image_omp               # Binario compilado (OpenMP)
    └── README.md
```

---

## Dependencias

- Compilador C++ con soporte para OpenMP (`g++`)
- Librería `libpng`

Instalación en Ubuntu/Debian:

```bash
sudo apt install libpng-dev
```

---

## Compilación

### Versión secuencial

```bash
cd secuencial/
g++ main.cpp -lpng -o image_seq
```

### Versión OpenMP

```bash
cd lab02/
g++ -fopenmp main.cpp -lpng -o image_omp
```

---

## Ejecución

### Versión secuencial

```bash
cd secuencial/
./image_seq ../images/small.png ../images/outputs/secuencial/out_small.png
```

### Versión OpenMP

```bash
cd lab02/
# Con N hilos
OMP_NUM_THREADS=N ./image_omp ../images/small.png ../images/outputs/lab02/out_small.png
```

---

## Resultados

La versión OpenMP reduce significativamente el tiempo de cómputo respecto a la versión secuencial, incluso con un único hilo. Sin embargo, el incremento del número de hilos no produce mejoras proporcionales debido a la naturaleza *memory-bound* del algoritmo y al peso de las operaciones de E/S en el tiempo total de ejecución.

---

## Repositorio

[https://github.com/CelesteLD/Image-Processing](https://github.com/CelesteLD/Image-Processing)