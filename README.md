# COMP 6630 – Commit Classification Project

## Overview
This project performs **multi-class commit type classification** using both **traditional machine learning** (TF-IDF + Logistic Regression, Naive Bayes, Random Forest) and **deep learning** (BERT) approaches. The goal is to predict the macro-type of a software commit from its natural-language commit message.

Key functionalities include:

- **Text preprocessing**: Lowercasing, stopword removal, URL & punctuation cleanup  
- **TF-IDF vectorization** for classical ML models  
- **BERT fine-tuning** for improved performance  
- **Model evaluation**: Accuracy, F1-score, per-class metrics  
- **Visualizations**:
  - Commit type distribution in the dataset
  - Commit message length distribution
  - Confusion matrices (TF-IDF & BERT)
  - Normalized BERT confusion matrix
  - Top TF-IDF features from Random Forest
  - Prediction distribution for BERT
- **Misclassified commit analysis**: Inspect incorrectly predicted commit messages  

---

## Dataset
The dataset used is [`0x404/ccs_dataset`](https://huggingface.co/datasets/0x404/ccs_dataset) from HuggingFace.

**Columns:**
- `Message`: Commit message text  
- `Ground truth`: Commit type label (`build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`)

**Splits:**
- `train`
- `eval`
- `test`

---

## Features
- **Preprocessing**: Clean commit messages by:
  - Lowercasing
  - Removing URLs
  - Removing non-alphabetic characters
  - Removing stopwords
- **TF-IDF Features**: Maximum 5000 features  
- **Models**:
  - Logistic Regression (balanced)
  - Multinomial Naive Bayes
  - Random Forest
  - BERT (`bert-base-uncased`) fine-tuned for 10 epochs
- **Evaluation Metrics**:
  - Accuracy
  - Macro F1-score
  - Confusion matrices
- **Visualizations**:
  - Commit type distribution
  - Message length histogram
  - Top TF-IDF features
  - Model prediction distribution
  - Confusion matrices for TF-IDF models
  - BERT normalized confusion matrix

---

## Prerequisites

- **Python 3.12 or higher**  
  Download and install from [python.org](https://www.python.org/downloads/)

- **Git** (optional, if cloning the repository)  
  Download from [git-scm.com](https://git-scm.com/downloads)

---

## Setup & Installation

1. **Clone the repository**
```bash
git clone git@github.com:kritichi9/commitanalysis.git
cd commit-classification
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
Required packages include: transformers, datasets, sentencepiece, torch, scikit-learn, nltk, matplotlib, seaborn, pandas, numpy.

4. **Download NLTK stopwords**
```bash
import nltk
nltk.download('stopwords')
```

5. **Running the Project**

You can run either as a Jupyter Notebook or a Python script:
Jupyter Notebook

```bash
jupyter notebook
```

Open CommitAnalysis.ipynb and run all cells. Also, the file with all output's is added as CommitAnalysiswithOutput.ipynb which can downloaded and viewed locally. 
Note: For some reason CommitAnalysiswithOutput.ipynb this is not viewable properly on git we need to download the file locally
For running through the Python Script. But it is better to run on Google Collab as it has GPU support and running the model locally is slow. Run:
``` python commit_analysis.py```

6. **Results**

Training, validation, and test performance are reported for all models

Confusion matrices and misclassification samples help understand model behavior

Top TF-IDF features and BERT predictions are visualized for interpretability

