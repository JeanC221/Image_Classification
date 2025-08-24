# CIFAR-10 Image Classification with PyTorch

Proyecto completo de clasificación de imágenes usando PyTorch y el dataset CIFAR-10. Este proyecto implementa una red neuronal convolucional simple para clasificar imágenes en 10 categorías diferentes.

##  Descripción

Este proyecto entrena una CNN simple para clasificar imágenes del dataset CIFAR-10 en las siguientes 10 categorías:
- Airplane
- Automobile  
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

##  Arquitectura del Modelo

### SimpleCNN
- **3 Capas Convolucionales**: 
  - Conv1: 3 → 32 canales (kernel 3x3)
  - Conv2: 32 → 64 canales (kernel 3x3)
  - Conv3: 64 → 128 canales (kernel 3x3)
- **Max Pooling** después de cada capa convolucional
- **2 Capas Fully Connected**:
  - FC1: 512 → 256 neuronas
  - FC2: 256 → 10 clases
- **Dropout** (0.5) para regularización
- **Activación ReLU** en todas las capas ocultas

##  Estructura del Proyecto

```
cifar10-classification/
│
├── model.py           # Definición de la arquitectura CNN
├── train.py           # Script principal de entrenamiento y evaluación  
├── utils.py           # Funciones auxiliares (visualización, métricas)
├── requirements.txt   # Dependencias del proyecto
├── README.md          # Este archivo
│
├── data/              # Dataset CIFAR-10 (se descarga automáticamente)
├── saved_models/      # Modelos entrenados guardados
└── outputs/           # Gráficas y resultados (opcional)
```

##  Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd cifar10-classification
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar entrenamiento
```bash
python train.py
```

##  Configuración

### Parámetros por defecto:
- **Batch size**: 32
- **Épocas**: 10
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **Data augmentation**: RandomHorizontalFlip, RandomRotation

### Personalizar entrenamiento:
Puedes modificar los parámetros en `train.py`:
```python
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
```

##  Resultados Esperados

Con la configuración por defecto, el modelo debería alcanzar:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~75-80%

### Métricas mostradas durante entrenamiento:
- Loss y accuracy por época
- Progreso por batches
- Tiempo de entrenamiento
- Matriz de confusión
- Reporte de clasificación detallado
- Ejemplos de predicciones vs etiquetas reales

##  Visualizaciones Incluidas

El proyecto genera automáticamente:
1. **Imágenes de ejemplo** del dataset
2. **Gráficas de entrenamiento** (loss y accuracy)
3. **Matriz de confusión**
4. **Predicciones vs etiquetas reales**
5. **Reporte de clasificación** por clase

##  Funcionalidades Principales

### model.py
- Clase `SimpleCNN` con arquitectura configurable
- Método `get_model_summary()` para información del modelo
- Forward pass optimizado

### train.py  
- Carga automática del dataset CIFAR-10
- Data augmentation para mejorar generalización
- Loop de entrenamiento con métricas detalladas
- Evaluación completa en conjunto de test
- Guardado automático del modelo entrenado

### utils.py
- `show_sample_images()`: Muestra ejemplos del dataset
- `plot_training_history()`: Gráfica de entrenamiento
- `show_predictions()`: Predicciones vs ground truth
- `calculate_accuracy()`: Cálculo de accuracy
- `plot_confusion_matrix()`: Matriz de confusión
- `get_classification_report()`: Reporte detallado
- `save_model()` / `load_model()`: Persistencia del modelo



##  Troubleshooting

### Error de CUDA:
```bash
# Instalar versión CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Memoria insuficiente:
```python
# Reducir batch size en train.py:
BATCH_SIZE = 16  
```

### Dataset no se descarga:
```python
# Descargar manualmente CIFAR-10 y colocar en ./data/
```

##  Monitoreo de Progreso

Durante el entrenamiento verás:
```
Epoch [1/10], Batch [200/1563], Loss: 1.8234, Acc: 32.15%
Epoch [1/10], Batch [400/1563], Loss: 1.5678, Acc: 45.23%
...
Epoch [1/10] COMPLETED:
  Average Loss: 1.4521
  Accuracy: 48.75%
  Time: 45.2s
```


##  Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE para detalles.

##  Autor

JeanC221
---

### 🚀 ¡Listo para empezar!

```bash
python train.py
```

El script se encargará de todo:
1. Descargar el dataset CIFAR-10
2. Crear y entrenar el modelo
3. Mostrar métricas y visualizaciones
4. Guardar el modelo entrenado
5. Evaluar en conjunto de test

