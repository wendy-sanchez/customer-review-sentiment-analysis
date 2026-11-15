# Análisis de Sentimiento de Reseñas de Clientes — WeLoveReviews

Proyecto de NLP que utiliza modelos Transformer preentrenados para analizar reseñas de clientes, clasificar sentimientos y comparar los comentarios escritos con las valoraciones numéricas.

[English](README.md) | [Español](README.es.md)

## Descripción general

Este proyecto explora cómo el Procesamiento del Lenguaje Natural (NLP) y los modelos de lenguaje preentrenados pueden ayudar a las empresas a comprender las opiniones de sus clientes más allá de las valoraciones numéricas.

Desarrollado a partir de un caso ficticio de cliente para WeLoveReviews, el proyecto analiza 500 reseñas de clientes de Harbor House Café.

El negocio tiene una valoración media de 4,5 sobre 5 estrellas. Sin embargo, el objetivo principal es determinar si el sentimiento expresado en las reseñas escritas refleja realmente este elevado nivel de satisfacción.

En lugar de entrenar un modelo desde cero, el proyecto integra modelos preentrenados de análisis de sentimiento de Hugging Face, evalúa sus predicciones, investiga errores de clasificación y transforma los resultados en conclusiones útiles para el negocio.

## Objetivos del proyecto

- Explorar y comprender la estructura y calidad de un conjunto de datos de reseñas de clientes.
- Realizar un Análisis Exploratorio de Datos (EDA) para identificar patrones y posibles problemas de calidad.
- Integrar un modelo preentrenado y multilingüe de análisis de sentimiento.
- Clasificar las reseñas de clientes en categorías de sentimiento positivo, neutral y negativo.
- Comparar las predicciones del modelo con las valoraciones originales de los clientes.
- Identificar e investigar predicciones de falsos negativos.
- Evaluar un modelo preentrenado alternativo para analizar posibles mejoras.
- Desarrollar un pipeline reutilizable en Python para automatizar el análisis de sentimiento.
- Comunicar los hallazgos mediante un Jupyter Notebook ejecutado y documentado.

## Conjunto de datos

El proyecto utiliza un conjunto de datos con 500 reseñas de clientes de Harbor House Café.

**Archivo de origen:** `data/raw/reviews.csv`

| Columna | Descripción |
|---|---|
| `review_id` | Identificador único de la reseña |
| `rating` | Valoración del cliente de 1 a 5 estrellas |
| `review_text` | Comentario escrito por el cliente |

### Hallazgos iniciales del conjunto de datos

- 500 registros y 3 columnas originales.
- No hay valores faltantes.
- Valoración media de los clientes: 4,50/5.
- El 86,4% de las reseñas tiene una valoración de 4 o 5 estrellas.
- El 6% de las reseñas tiene una valoración de 1 o 2 estrellas.
- La longitud de los textos oscila entre 86 y 231 caracteres.
- Se identificaron siete textos de reseñas repetidos.
- Se detectó un problema de codificación de texto en 433 registros.

El análisis exploratorio confirmó que el conjunto de datos estaba completo, pero presentaba un claro desequilibrio hacia las valoraciones positivas.

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)

El análisis inicial incluye:

- Dimensiones y tipos de datos del conjunto.
- Detección de valores faltantes.
- Distribución de valoraciones y estadísticas descriptivas.
- Análisis de la longitud de los textos de las reseñas.
- Muestras de reseñas de distintas categorías de valoración.
- Detección de duplicados e inspección de la codificación de texto.

### 2. Clasificación de sentimientos

El modelo principal utilizado es:

`nlptown/bert-base-multilingual-uncased-sentiment`

Este modelo preentrenado basado en BERT predice el sentimiento mediante una escala de cinco estrellas.

Sus predicciones se transforman en tres categorías de sentimiento orientadas al negocio:

| Valoración predicha | Sentimiento |
|---|---|
| 1–2 estrellas | Negativo |
| 3 estrellas | Neutral |
| 4–5 estrellas | Positivo |

El modelo y el tokenizador se cargan una sola vez antes de ejecutar la inferencia para evitar cargas repetidas innecesarias.

Las predicciones incluyen la valoración estimada, la categoría de sentimiento y la puntuación de confianza del modelo.

### 3. Evaluación del modelo y análisis de errores

Las predicciones del modelo se comparan con las valoraciones originales de los clientes.

Se presta especial atención a los falsos negativos: reseñas con valoraciones positivas de los clientes que el modelo clasifica incorrectamente como negativas.

El análisis investiga el desajuste de dominio como posible causa, ya que el modelo principal fue ajustado con reseñas de productos, mientras que el conjunto de datos contiene reseñas de servicios.

También se realiza una revisión manual de 15 ejemplos seleccionados para evaluar la calidad de las predicciones.

### 4. Comparación con un modelo alternativo

Un experimento opcional evalúa un segundo modelo preentrenado:

`tabularisai/multilingual-sentiment-analysis`

Ambos modelos se evalúan utilizando las mismas 500 reseñas.

La comparación analiza el número de falsos negativos y la concordancia entre las categorías de sentimiento predichas y las categorías derivadas de las valoraciones de los clientes.

El modelo alternativo tiene una licencia CC-BY-NC-4.0 de uso no comercial, que debe tenerse en cuenta antes de cualquier utilización comercial.

## Resultados y hallazgos principales

### Modelo principal: nlptown

El análisis de sentimiento se completó sobre las 500 reseñas.

| Sentimiento | Reseñas | Porcentaje |
|---|---:|---:|
| Positivo | 408 | 81,6% |
| Neutral | 41 | 8,2% |
| Negativo | 51 | 10,2% |

| Métrica | Resultado |
|---|---:|
| Valoración media humana | 4,50 |
| Valoración media predicha | 4,36 |
| Diferencia | 0,14 estrellas |
| Falsos negativos | 20 |
| Concordancia de categorías | 93,8% |

Los resultados muestran que aproximadamente una de cada diez reseñas escritas se clasifica como negativa, a pesar de que el negocio tiene una valoración media elevada.

Esto demuestra por qué los promedios numéricos por sí solos pueden ocultar señales importantes de insatisfacción de los clientes.

### Modelo alternativo: tabularisai

El modelo alternativo produjo la siguiente comparación:

| Métrica | Modelo principal | Modelo alternativo |
|---|---:|---:|
| Falsos negativos | 20 | 3 |
| Concordancia de categorías | 93,8% | 91,4% |

El modelo alternativo redujo los falsos negativos en un 85%, identificando menos reseñas con valoraciones positivas que habían sido clasificadas como negativas.

Sin embargo, su concordancia general entre categorías de sentimiento fue ligeramente inferior.

Esto pone de manifiesto una cuestión importante del aprendizaje automático: mejorar un tipo de error de clasificación no implica necesariamente mejorar todas las métricas de evaluación.

## Pipeline de Python

El proyecto incluye un pipeline de inferencia reutilizable en `src/app.py`.

Sus principales responsabilidades son:

1. Cargar el modelo preentrenado de análisis de sentimiento.
2. Leer las reseñas de clientes desde el archivo CSV original.
3. Ejecutar la inferencia de sentimiento.
4. Transformar las valoraciones predichas en categorías de sentimiento orientadas al negocio.
5. Calcular la longitud del texto, la valoración predicha y las puntuaciones de confianza.
6. Exportar el conjunto de datos enriquecido a un archivo CSV procesado.
7. Mostrar un resumen de la distribución de sentimientos y de la comparación de valoraciones.

### Archivo de salida generado

El pipeline genera:

`data/processed/reviews_with_sentiment.csv`

El archivo de salida contiene la información original de las reseñas y cuatro columnas adicionales:

| Columna | Descripción |
|---|---|
| `text_len` | Longitud del texto de la reseña |
| `pred_stars` | Valoración predicha por el modelo |
| `pred_score` | Puntuación de confianza del modelo |
| `sentiment` | Clasificación: positivo, neutral o negativo |

El pipeline puede ejecutarse nuevamente cuando haya nuevas reseñas disponibles, siempre que respeten la estructura de datos esperada.

## Stack tecnológico

| Tecnología | Uso |
|---|---|
| Python | Lenguaje de programación principal |
| Pandas | Manipulación y análisis de datos |
| NumPy | Operaciones numéricas |
| Matplotlib | Visualización de datos |
| Seaborn | Visualización estadística |
| Hugging Face Transformers | Modelos NLP preentrenados e inferencia |
| BERT | Arquitectura principal para el análisis de sentimiento |
| DistilBERT | Arquitectura del modelo alternativo |
| PyTorch | Framework de aprendizaje profundo |
| SQLAlchemy | Conectividad opcional con bases de datos |
| PostgreSQL | Conexión opcional con bases de datos relacionales |
| Jupyter Notebook | Análisis exploratorio y documentación técnica |
| GitHub | Control de versiones y documentación del proyecto |

## Estructura del repositorio

```text
aie-pt-3--machine-learning-python/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   └── processed/
│       └── reviews_with_sentiment.csv  # Archivo generado
│
├── src/
│   ├── explore.ipynb
│   ├── app.py
│   └── utils.py
│
├── models/
├── .devcontainer/
├── .vscode/
├── .env.example
├── requirements.txt
├── CONTEXT.md
├── README.md
└── README.es.md
```

## Cómo empezar

### Requisitos previos

- Python 3.11 o superior.
- pip.
- Conexión a Internet para descargar los modelos preentrenados desde Hugging Face.

### Instalación

Clonar el repositorio y acceder a su directorio.

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### Ejecución del análisis

Para explorar el conjunto de datos y consultar el flujo analítico completo, abrir:

`src/explore.ipynb`

Se puede utilizar Jupyter Notebook, JupyterLab o el soporte de notebooks de Visual Studio Code.

Para ejecutar el pipeline reutilizable de análisis de sentimiento desde la raíz del repositorio:

```bash
python src/app.py
```

La primera ejecución descarga el modelo preentrenado y el tokenizador necesarios desde Hugging Face. El archivo CSV enriquecido con las predicciones de sentimiento se guarda en `data/processed/`.

### Configuración opcional de la base de datos

El proyecto incluye un módulo auxiliar de conexión a PostgreSQL en `src/utils.py`.

Si se necesita conectividad con una base de datos, crear un archivo `.env` en la raíz del repositorio y configurar:

```env
DATABASE_URL=postgresql://USUARIO:CONTRASEÑA@HOST:PUERTO/NOMBRE_BD
```

La conexión a la base de datos no es necesaria para ejecutar el pipeline de análisis de sentimiento basado en CSV. La aplicación continúa procesando el conjunto de datos aunque la conexión opcional no esté disponible.

## Alcance y limitaciones del proyecto

- El proyecto utiliza un conjunto de datos de ejemplo de 500 reseñas, no una integración en directo con opiniones de clientes.
- El modelo principal fue ajustado originalmente con reseñas de productos, lo que puede afectar a su rendimiento con reseñas relacionadas con servicios.
- Las predicciones del modelo son estimaciones automatizadas de sentimiento y no deben sustituir la revisión humana en casos ambiguos.
- El modelo alternativo se utiliza como experimento comparativo y tiene una licencia de uso no comercial.
- El CSV generado está pensado para alimentar posteriores procesos de análisis e informes, no es una aplicación desplegada para clientes.

---

**Developer Wendy Sanchez**
