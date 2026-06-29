# Hassia LAT

Aplicación web que usa inteligencia artificial para diagnosticar el estado de salud
de la palta Hass (hojas y frutos) cultivada en el norte del Perú. El usuario sube una
foto y la app indica si está **Sana** o **Enferma**, con un porcentaje de confianza y
una recomendación de manejo en español.

## Fase actual

Fase 1: clasificación de una sola foto subida por el usuario. El módulo de
clasificación (`app/inference.py`) está desacoplado de Flask para que una futura
fase de detección en video pueda llamarlo por recorte de cuadro sin reescribir el
backend.

## Dataset

Clasificación binaria (`Disease` / `Healthy`) — el dataset original no trae
etiquetas por enfermedad/plaga específica, solo sano vs. enfermo. Ubicado en
`data/Avocado Augmneted_Dataset/{train,valid,test}/{Disease,Healthy}`.

## Estructura del proyecto

```
config.py              # rutas y parámetros compartidos
training/
  dataset.py            # carga de datasets train/valid/test
  model.py               # backbone preentrenado + cabeza de clasificación
  train.py               # entrenamiento, métricas y matriz de confusión
app/
  __init__.py            # fábrica de la app Flask
  routes.py               # endpoints / y /predict
  inference.py            # clasificador reutilizable (PIL.Image -> predicción)
  recommendations.py      # nombres en español y recomendaciones por clase
  templates/index.html    # interfaz en español
run.py                  # punto de entrada de la app
```

## Entrenamiento

```
pip install -r requirements-training.txt
python training/train.py
python training/export_tflite.py
```

Genera en `models/`: `hassia_model.keras`, `hassia_model.tflite`,
`class_names.json`, `confusion_matrix.png`, `training_history.png` y
`classification_report.txt`.

La app Flask solo usa `hassia_model.tflite` (vía `requirements.txt`, sin
TensorFlow completo) — esto evita quedarse sin memoria en planes gratuitos
de hosting como Render (512MB RAM), donde cargar TensorFlow + Keras
completo agotaba el límite.

## Ejecutar la app

```
pip install -r requirements.txt
python run.py
```

Abrir `http://localhost:5000`.
