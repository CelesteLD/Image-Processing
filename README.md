# Lab 2 — Procesamiento de Imágenes con OpenMP

## Descripción

Este proyecto implementa un algoritmo de **procesamiento de imágenes en C++**, aplicando un filtro de convolución sobre imágenes en formato PNG.

El objetivo principal es analizar el impacto del **paralelismo con OpenMP** en el rendimiento del procesamiento, comparando la ejecución secuencial con versiones paralelas.

Se ha sustituido el filtro Gaussiano original por un **filtro Sharpen**, que permite resaltar bordes y detalles de la imagen.

---

## Tecnologías utilizadas

* C++
* OpenMP
* png++ (para manejo de imágenes PNG)

---

## 📂 Estructura del proyecto

```
lab2-cn/
├── main.cpp
├── images/
│   ├── small.png
│   ├── medium.png
│   └── large.png
├── .gitignore
└── README.md
```

---

## Filtro aplicado

Se ha implementado un filtro **Sharpen**, definido por la siguiente máscara:

```
 0  -1   0
-1   5  -1
 0  -1   0
```

Este filtro permite aumentar la nitidez de la imagen, resaltando bordes y detalles.

---

## Compilación

Es necesario tener instaladas las librerías `libpng` y `png++`.

```bash
sudo apt update
sudo apt install libpng-dev libpng++-dev
```

Compilar el programa:

```bash
g++ main.cpp -lpng -fopenmp -O2
```

---

## Ejecución

El programa requiere dos argumentos:

```bash
./a.out <imagen_entrada.png> <imagen_salida.png>
```

Ejemplo:

```bash
./a.out images/large.png images/outputs/out_large.png
```

---

## Paralelización

Se ha paralelizado el algoritmo de convolución utilizando OpenMP:

```cpp
#pragma omp parallel for collapse(3)
```

Esto permite distribuir el cálculo de cada píxel entre múltiples hilos, ya que cada operación es independiente.

---

## Análisis de rendimiento

Se han realizado pruebas con imágenes de diferentes tamaños:

* Pequeña (~256×256)
* Mediana (~512×512)
* Grande (~1920×1080)

Y con distinto número de hilos:

* 1 hilo (secuencial)
* 2 hilos
* 4 hilos
* 8 hilos

### Observaciones

* Para imágenes pequeñas, el paralelismo no mejora el rendimiento debido al overhead.
* Para imágenes grandes, se observa una mejora significativa del tiempo de ejecución.
* El speedup no es lineal debido a:

  * Costes de gestión de hilos
  * Acceso a memoria
  * Limitaciones de hardware

---

## Conclusión

La paralelización mediante OpenMP resulta especialmente efectiva en problemas altamente paralelizables como la convolución de imágenes, donde cada píxel puede procesarse de forma independiente.

El rendimiento mejora con el tamaño de la imagen, evidenciando que el paralelismo es más eficiente cuando la carga computacional es elevada.

---

## Notas adicionales

* Las imágenes deben estar en formato PNG.
* Se recomienda usar ImageMagick para convertir imágenes:

```bash
convert imagen.jpg imagen.png
```

---

## Autora

Celeste Luis — Máster en Ingeniería Informática
Proyecto académico — Computación en la Nube
