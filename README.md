# COMP 6630 - Commit Classification Project

## Overview
This project performs **multi-class commit type classification** using machine learning techniques. Rare commit types are merged into an "Other" category, and TF-IDF features are extracted from commit messages. Two models are implemented:

- Logistic Regression (with class balancing)
- Multinomial Naive Bayes

The project also includes:
- Dataset exploration and visualization
- Confusion matrices and per-class metrics
- Top words per commit type
- Misclassified commit analysis

---

## Dataset
The dataset used is `meriemm6/commit-classification-dataset` from the HuggingFace Hub.

**Columns:**
- `Message`: Commit message text
- `Ground truth`: Commit type label

---

## Features
- **Text preprocessing**: Lowercasing, stopword removal, URL & punctuation cleanup
- **TF-IDF vectorization** for feature extraction
- **Model evaluation**: Accuracy, F1-score, per-class metrics
- **Visualizations**:
  - Commit type distribution
  - Message length distribution
  - Confusion matrices
  - Top words per commit type

---

## Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd commit-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebook or script:
```bash
python commit_classification.py
```

