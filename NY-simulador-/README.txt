El proyecto está compuesto por tres archivos principales en Python:

1)selección_modelos.py
En este script se prueban distintos modelos de Machine Learning (LightGBM, Random Forest, XGBoost, etc.) y se comparan sus métricas.
El objetivo es seleccionar el modelo con mejor rendimiento.
Ejecución no obligatoria, pero muestra el proceso de selección del modelo.

2)entrenar_modelo.py
Este archivo entrena el modelo elegido en el paso anterior y guarda el modelo entrenado en el archivo: modelo_simulador.pk
la ejecución del archivo no es obligatoria antes de lanzar la aplicación ya que se incluirá el archivo  modelo_simulador.pk en el zip.
Si se ejecutará el modelo puede guardarse en una carpeta diferente, por lo que sería necesario mover manualmente el archivo a la carpeta donde se encuentra el archivo practica.py y eliminar el adjuntado.

3)practica.py
Contiene la aplicación principal en Streamlit, que permite al usuario interactuar con los filtros, visualizar resultados en tablas y mapas 3D, y consultar la predicción del modelo entrenado.

Para que ningún archivo de error es probable que se necesite la instalación de varias librerías ya que son de uso poco convencional.