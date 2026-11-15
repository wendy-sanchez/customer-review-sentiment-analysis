# Customer Review Sentiment Analysis — WeLoveReviews

An NLP project using pretrained Transformer models to analyze customer reviews, classify sentiment, and compare written feedback against numerical ratings.

[English](README.md) | [Español](README.es.md)

## Overview

This project explores how Natural Language Processing (NLP) and pretrained language models can help businesses understand customer feedback beyond numerical ratings.

Developed around a fictional client scenario for WeLoveReviews, the project analyzes 500 customer reviews from Harbor House Café.

The business has an average customer rating of 4.5 out of 5 stars. However, the main objective is to determine whether the sentiment expressed in written reviews actually reflects this high satisfaction score.

Instead of training a model from scratch, the project integrates pretrained sentiment analysis models from Hugging Face, evaluates their predictions, investigates classification errors, and translates the results into actionable business insights.

## Project Objectives

- Explore and understand the structure and quality of a customer review dataset.
- Perform Exploratory Data Analysis (EDA) to identify patterns and potential data quality issues.
- Integrate a pretrained multilingual sentiment analysis model.
- Classify customer reviews into positive, neutral, and negative sentiment categories.
- Compare model predictions against customers' original star ratings.
- Identify and investigate false negative predictions.
- Evaluate an alternative pretrained model to assess potential improvements.
- Develop a reusable Python pipeline to automate sentiment analysis.
- Communicate findings through an executed Jupyter Notebook.

## Dataset

The project uses a dataset containing 500 customer reviews from Harbor House Café.

**Source file:** `data/raw/reviews.csv`

| Column | Description |
|---|---|
| `review_id` | Unique review identifier |
| `rating` | Customer rating from 1 to 5 stars |
| `review_text` | Written customer feedback |

### Initial Dataset Insights

- 500 records and 3 original columns.
- No missing values.
- Average customer rating: 4.50/5.
- 86.4% of reviews have a rating of 4 or 5 stars.
- 6% of reviews have a rating of 1 or 2 stars.
- Review text length ranges from 86 to 231 characters.
- Seven repeated review texts were identified.
- A text encoding issue was detected in 433 records.

The exploratory analysis confirmed that the dataset was complete but highly imbalanced toward positive ratings.

## Methodology

### 1. Exploratory Data Analysis (EDA)

The initial analysis includes:

- Dataset dimensions and data types.
- Missing value detection.
- Rating distribution and descriptive statistics.
- Review text length analysis.
- Review samples across rating categories.
- Duplicate detection and text encoding inspection.

### 2. Sentiment Classification

The primary model is:

`nlptown/bert-base-multilingual-uncased-sentiment`

This pretrained BERT-based model predicts sentiment on a five-star scale.

Its predictions are mapped into three business-oriented sentiment categories:

| Predicted Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

The model and tokenizer are loaded once before inference to avoid unnecessary repeated loading.

Predictions include the estimated rating, sentiment category, and model confidence score.

### 3. Model Evaluation & Error Analysis

The model's predictions are compared against the original customer ratings.

A particular focus is placed on false negatives: reviews with positive customer ratings that the model incorrectly classifies as negative.

The analysis investigates domain mismatch as a possible cause, since the primary model was fine-tuned on product reviews while the dataset contains service reviews.

A manual review of 15 selected examples is also performed to assess prediction quality.

### 4. Alternative Model Comparison

An optional experiment evaluates a second pretrained model:

`tabularisai/multilingual-sentiment-analysis`

Both models are evaluated using the same 500 reviews.

The comparison examines false negative counts and agreement between predicted sentiment categories and categories derived from customer ratings.

The alternative model has a non-commercial CC-BY-NC-4.0 license, which must be considered before any commercial use.

## Results & Key Findings

### Primary Model: nlptown

<badge color="success">Inference completed · 500 reviews</badge>

| Sentiment | Reviews | Percentage |
|---|---:|---:|
| Positive | 408 | 81.6% |
| Neutral | 41 | 8.2% |
| Negative | 51 | 10.2% |

| Metric | Result |
|---|---:|
| Average human rating | 4.50 |
| Average predicted rating | 4.36 |
| Difference | 0.14 stars |
| False negatives | 20 |
| Sentiment-band agreement | 93.8% |

The results show that approximately one in ten written reviews is classified as negative, despite the business having a high average rating.

This demonstrates why numerical averages alone may hide important customer dissatisfaction signals.

### Alternative Model: tabularisai

The alternative model produced the following comparison:

| Metric | Primary Model | Alternative Model |
|---|---:|---:|
| False negatives | 20 | 3 |
| Sentiment-band agreement | 93.8% | 91.4% |

The alternative model reduced false negatives by 85%, identifying fewer positively rated reviews as negative.

However, its overall sentiment-band agreement was slightly lower.

This highlights an important machine learning trade-off: improving one type of classification error does not necessarily improve every evaluation metric.

## Python Pipeline

The project includes a reusable inference pipeline in `src/app.py`.

Its main responsibilities are:

1. Load the pretrained sentiment analysis model.
2. Read customer reviews from the raw CSV dataset.
3. Run sentiment inference.
4. Convert predicted star ratings into business sentiment categories.
5. Calculate text length, predicted rating, and confidence scores.
6. Export the enriched dataset to a processed CSV file.
7. Print a summary of sentiment distribution and rating comparisons.

### Generated Output

The pipeline generates:

`data/processed/reviews_with_sentiment.csv`

The output contains the original review information and four additional columns:

| Column | Description |
|---|---|
| `text_len` | Review text length |
| `pred_stars` | Model-predicted star rating |
| `pred_score` | Model confidence score |
| `sentiment` | Positive, Neutral, or Negative classification |

The pipeline can be executed again when new reviews become available, provided they follow the expected dataset structure.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Hugging Face Transformers | Pretrained NLP models and inference |
| BERT | Primary sentiment analysis architecture |
| DistilBERT | Alternative model architecture |
| PyTorch | Deep learning framework |
| SQLAlchemy | Optional database connectivity |
| PostgreSQL | Optional relational database connection |
| Jupyter Notebook | Exploratory analysis and technical documentation |
| GitHub | Version control and project documentation |

## Repository Structure

```text
aie-pt-3--machine-learning-python/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   └── processed/
│       └── reviews_with_sentiment.csv  # Generated output
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

## Getting Started

### Prerequisites

- Python 3.11 or later.
- pip.
- Internet connection to download pretrained models from Hugging Face.

### Installation

Clone the repository and navigate to its directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Analysis

To explore the dataset and review the complete analytical workflow, open:

`src/explore.ipynb`

You can use Jupyter Notebook, JupyterLab, or the notebook support in Visual Studio Code.

To execute the reusable sentiment analysis pipeline from the repository root:

```bash
python src/app.py
```

The first execution downloads the required pretrained model and tokenizer from Hugging Face. The generated sentiment-enriched CSV is saved under `data/processed/`.

### Optional Database Configuration

The project includes an optional PostgreSQL connection helper in `src/utils.py`.

If database connectivity is required, create a `.env` file in the repository root and configure:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
```

Database connectivity is not required for the CSV-based sentiment inference pipeline. The application continues processing the dataset if the optional database connection is unavailable.

## Project Scope & Limitations

- The project uses a sample dataset of 500 reviews rather than a live customer feedback integration.
- The primary model was originally fine-tuned on product reviews, which may affect its performance on service-related reviews.
- Model predictions represent automated sentiment estimates and should not replace human review in ambiguous cases.
- The alternative model is an experimental comparison and has a non-commercial license.
- The generated CSV is intended as an input for further reporting and analysis, not as a deployed customer-facing application.

---

**Developer Wendy Sanchez**
