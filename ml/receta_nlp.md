
# 🧪 Receta para Modelos de NLP: Una Guía Práctica para Científicos de Datos

## 📚 Ingredientes Básicos

### Herramientas Fundamentales
- **Python 3.8+**: Nuestro lenguaje base
- **IDE**: VSCode o PyCharm con soporte para Jupyter
- **Control de versiones**: Git
- **Entorno virtual**: conda o venv

### Librerías Esenciales
- **Procesamiento de datos**: pandas, numpy
- **NLP**: spaCy, NLTK, transformers (Hugging Face)
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch o TensorFlow
- **Visualización**: matplotlib, seaborn

### Recursos Computacionales
- Mínimo: 16GB RAM, CPU multi-core
- Recomendado: GPU NVIDIA con 8GB+ VRAM
- Alternativa: Google Colab Pro o AWS SageMaker

## 👩‍🍳 Preparación (Pre-procesamiento)

### 1. Recolección de Datos
**¿Qué?** Obtener un dataset limpio y representativo.

**¿Cómo?**
1. Identifica fuentes confiables (Kaggle, UCI, datos propios)
2. Establece criterios de calidad
3. Define el formato de los datos

**Herramientas:**
```python
import pandas as pd
from datasets import load_dataset  # Hugging Face datasets

# Ejemplo de carga
dataset = load_dataset('csv', data_files='datos.csv')
# o
df = pd.read_csv('datos.csv')
```

**Documentación:**
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### 2. Limpieza y Normalización
**¿Qué?** Preparar el texto para su procesamiento.

**¿Cómo?**
1. Eliminar ruido (caracteres especiales, HTML)
2. Normalizar texto (minúsculas, acentos)
3. Tokenización
4. Eliminar stopwords (opcional)

**Herramientas:**
```python
import spacy
import re
from unidecode import unidecode

nlp = spacy.load('es_core_news_sm')

def limpiar_texto(texto):
    texto = re.sub(r'<[^>]+>', '', texto)  # Eliminar HTML
    texto = unidecode(texto.lower())  # Normalizar
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)
```

**Documentación:**
- [spaCy Course](https://course.spacy.io/)
- [Regular Expressions](https://docs.python.org/3/library/re.html)

## 🥘 Preparación del Modelo

### 3. Vectorización del Texto
**¿Qué?** Convertir texto en vectores numéricos.

**¿Cómo?**
1. Elegir método de vectorización:
   - Bag of Words
   - TF-IDF
   - Word Embeddings
   - Embeddings contextuales

**Herramientas:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(textos)

# BERT Embeddings
tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
model = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
```

**Documentación:**
- [Scikit-learn: Vectorización de texto](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Transformers Documentation](https://huggingface.co/transformers/)

### 4. División y Evaluación
**¿Qué?** Preparar conjuntos de entrenamiento y prueba.

**¿Cómo?**
1. Dividir datos (típicamente 80/20)
2. Establecer métricas de evaluación
3. Implementar validación cruzada

**Herramientas:**
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## 🍳 Cocción (Entrenamiento)

### 5. Selección y Entrenamiento del Modelo
**¿Qué?** Elegir y entrenar el modelo adecuado.

**¿Cómo?**
1. Seleccionar arquitectura según tarea:
   - Clasificación: BERT, RoBERTa
   - Secuencias: LSTM, Transformer
   - Generación: GPT, T5
2. Ajustar hiperparámetros
3. Implementar early stopping

**Herramientas:**
```python
from transformers import Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression

# Modelo simple
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Modelo transformers
training_args = TrainingArguments(
    output_dir="./resultados",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)
```

### 6. Optimización y Ajuste
**¿Qué?** Mejorar el rendimiento del modelo.

**¿Cómo?**
1. Analizar errores comunes
2. Ajustar hiperparámetros
3. Probar técnicas de regularización
4. Implementar técnicas de data augmentation

**Herramientas:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(
    LogisticRegression(), 
    param_grid, 
    cv=5
)
grid_search.fit(X_train, y_train)
```

## 🍽️ Servido (Deployment)

### 7. Evaluación Final y Documentación
**¿Qué?** Validar y documentar el modelo.

**¿Cómo?**
1. Evaluar en conjunto de prueba
2. Documentar decisiones y resultados
3. Crear guías de uso
4. Establecer líneas base de rendimiento

**Herramientas:**
```python
import mlflow
import pickle

# Guardar modelo
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Logging con MLflow
mlflow.log_param("vectorizer", "tfidf")
mlflow.log_metric("accuracy", accuracy)
```

### 8. Despliegue y Monitoreo
**¿Qué?** Poner el modelo en producción.

**¿Cómo?**
1. Containerizar (Docker)
2. Crear API (FastAPI/Flask)
3. Implementar monitoreo
4. Establecer pipeline de actualización

**Herramientas:**
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    processed_text = limpiar_texto(text)
    vector = vectorizer.transform([processed_text])
    prediction = modelo.predict(vector)
    return {"prediccion": prediction.tolist()}
```

## 📚 Fuentes Recomendadas

1. Libros:
   - "Natural Language Processing with Transformers" - Lewis Tunstall
   - "Speech and Language Processing" - Daniel Jurafsky
   - "Natural Language Processing with Python" - Steven Bird

2. Recursos en línea:
   - [Hugging Face Documentation](https://huggingface.co/docs)
   - [spaCy Course](https://course.spacy.io)
   - [Papers with Code - NLP](https://paperswithcode.com/area/natural-language-processing)
   - [Medium - Towards Data Science](https://towardsdatascience.com/)

## 🔍 Notas Finales

- Mantén un registro detallado de experimentos
- Implementa control de versiones desde el inicio
- Considera aspectos éticos y sesgos en los datos
- Actualiza regularmente las dependencias
- Mantén documentación clara y actualizada

Recuerda: El éxito en NLP no solo depende de la implementación técnica, sino también de la comprensión profunda del problema y los datos. ¡Experimenta, itera y aprende de cada proyecto!
