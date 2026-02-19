# LaLiga – Optimización de Jornadas con Algoritmo Genético

Aplicación desarrollada en Python + Streamlit para optimizar una jornada de LaLiga utilizando un algoritmo genético. El objetivo es encontrar la jornada ideal maximizando determinados criterios estadísticos definidos en la función de fitness.

##  Descripción del Proyecto

La aplicación permite introducir manualmente una jornada de liga o generarla de forma aleatoria, optimizarla mediante un algoritmo genético, visualizar métricas estadísticas de la solución obtenida, mostrar gráficos comparativos y acceder a una simulación completa de una liga con sus 38 jornadas.

El proyecto combina optimización heurística, análisis combinatorio, complejidad computacional y visualización interactiva.

##  Fundamento Teórico

En el documento PDF incluido se desarrollan el cálculo del número total de combinaciones posibles de jornadas, el análisis de complejidad computacional, la estimación del orden de crecimiento (Big O) del algoritmo y la justificación del uso de algoritmos genéticos frente a una búsqueda exhaustiva.

##  Estructura del Proyecto

- laliga.py → Aplicación principal de Streamlit. Implementa el algoritmo genético y la interfaz interactiva.
- liga.pdf → Documento con el análisis combinatorio y complejidad Big O.
- pages/1_page.py → Página auxiliar accesible desde la barra lateral que simula una liga completa con sus 38 jornadas.

##  Funcionamiento

1) Generación de jornada: El usuario puede introducir manualmente los enfrentamientos o generar una jornada aleatoria inicial.

2) Optimización: Se aplica un algoritmo genético que incluye población inicial, selección, cruce, mutación y evaluación mediante función de fitness para maximizar la calidad estadística de la jornada.

3)  Visualización: La aplicación muestra métricas estadísticas, gráficos alternativos y resultados estructurados en una interfaz interactiva.

## Tecnologías Utilizadas

Python, Streamlit, NumPy, Pandas y Matplotlib.

## Complejidad Computacional

Dado el enorme espacio combinatorio de posibles jornadas, una búsqueda exhaustiva resulta inviable. Por ello se implementa un algoritmo genético cuyo coste aproximado es:

O(P · G · n)

donde P es el tamaño de población, G el número de generaciones y n la dimensión del problema. El análisis detallado se encuentra en el documento PDF adjunto.

## Ejecución

Para ejecutar la aplicación:
1) activar el entorno
 .\venv\Scripts\actívate
2) ejecutar el código principal(laliga-py)
 streamlit run laliga.py

