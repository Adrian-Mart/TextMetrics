# TextMetrics

TextMetrics es una herramienta para el análisis y predicción de ventas de videojuegos basada en similitud ponderada y optimización mediante un Micro Algoritmo Genético (MicroGA) con validación cruzada. El proyecto aborda el reto de predecir ventas utilizando modelos que consideran la distancia entre textos y parámetros ajustables, permitiendo categorizar y analizar datos de manera eficiente.

## Características principales

- **Modelo de similitud ponderada:** Evalúa la cercanía entre títulos de videojuegos para predecir ventas.
- **Optimización con MicroGA:** Ajuste automático de parámetros clave (generaciones, tamaño de población, tasa de mutación, coeficientes Alpha, Beta y Gamma) a través de 50 experimentos controlados.
- **Validación cruzada:** Uso de múltiples folds para asegurar la robustez y generalización del modelo.
- **Arquitectura adaptable:** Puede aplicarse a otras tareas de análisis textual como análisis de sentimientos, clasificación de documentos y detección de noticias falsas.


1. **Descargar y preparar datos:**

    ```bash
    python -m scripts.sets_builder.set_builder -sad 'a_titles.txt' -sbd 'b_titles.txt' -oa 'data_a' -ob 'data_b' -b 'demo_data'
    ```
    > Requiere listas de títulos (`a_titles.txt`, `b_titles.txt`) para crear la base de datos.

2. **Entrenar y probar el modelo:**

    ```bash
    python -m scripts.tunning.tunning -g 50 -p 500 -f 5 -zp 0.8 -m 0.5
    ```
    > Ajusta los parámetros según sea necesario para experimentar con diferentes configuraciones.

> Puedes usar las banderas `--help` o `-h` en cualquiera de los comandos para ver la lista completa de opciones y su uso.


## Aplicaciones

Aunque el enfoque principal es la predicción de ventas de videojuegos, la arquitectura de TextMetrics es flexible y puede adaptarse a:

- Análisis de sentimientos
- Clasificación de documentos
- Detección de noticias falsas

## Créditos

Desarrollado como parte de un estudio experimental sobre análisis y predicción en procesamiento de texto.

## Agradecimientos

Este proyecto utiliza datos descargados desde Wikipedia a través de su API. Agradecemos a Wikipedia por proporcionar acceso abierto a su base de datos, lo que ha sido fundamental para el desarrollo de TextMetrics.