# 📦 mls-utils

`mls-utils` es un módulo auxiliar diseñado para proyectos de scoring y segmentación de clientes, como el caso de Montepío. Este paquete permite generar datos simulados, calcular puntuaciones ponderadas, segmentar clientes con K-Means, mostrar recomendaciones personalizadas y visualizar los resultados con Gradio.

## 🚀 Funcionalidades

- `generate_mls_dataframe`: Genera un DataFrame simulado con datos demográficos, transaccionales y de comportamiento digital.
- `compute_weighted_score`: Calcula la suma ponderada de columnas especificadas (score transaccional o digital).
- `segment_and_profile`: Segmenta clientes usando K-Means y genera perfiles promedio por cluster.
- `plot_clusters`: Visualiza los clusters en un gráfico 2D con centroides resaltados.
- `quick_eda`: Realiza un pequeño análisis exploratorio de los datos.
- `mock_recommendation`: Muestra recomendaciones personalizadas basadas en el cluster del cliente.
- `predict_scores`: Simula la predicción de scores a partir de datos de un nuevo cliente.
- `launch_gradio_interface`: Lanza una interfaz Gradio para probar la recomendación y predicción de scores.

## 🔧 Instalación

### 1. Instalar desde GitHub (repositorio privado)

En tu notebook de Google Colab o entorno Python:

```python
!pip install git+https://<your-username>:<your-token>@github.com/Charles9412/mls-utils.git
