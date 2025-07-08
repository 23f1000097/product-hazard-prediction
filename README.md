# Product-hazard-prediction
# Product Hazard and Category Prediction


This project is a complete machine learning pipeline for predicting the **hazard type** and **product category** of consumer goods based on textual incident reports and structured data. It leverages a combination of Natural Language Processing (NLP) and a powerful gradient boosting model to achieve high-accuracy multi-label classification.

## Project Overview

The core task is to analyze incident reports—each containing a title, a detailed text description, and metadata like the date and country—and assign two labels to each report:
1.  **Hazard Type:** The specific danger the product poses (e.g., 'Choking Hazard', 'Fire Hazard').
2.  **Product Category:** The classification of the product itself (e.g., 'Toys', 'Electronics').

To solve this, two separate LightGBM models are trained on features derived from both the text and structured data.

### Key Features
*   **Multi-Label Classification:** Predicts two independent target variables from a single set of input features.
*   **Hybrid Feature Engineering:** Combines structured data (date, country) with sophisticated NLP features extracted from text.
*   **TF-IDF for Text Representation:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert raw text into meaningful numerical vectors, capturing important keywords and phrases.
*   **High-Performance Modeling:** Employs LightGBM, a fast and efficient gradient boosting framework known for its excellent performance.
*   **Class Imbalance Handling:** Implements class weights to ensure the model gives appropriate attention to rare categories, preventing bias towards the majority classes.

## The Datasets

The project uses two primary data files:
*   `Hazards_LABELLED_TRAIN.csv`: A training set containing 5,082 records with all features and the correct `hazard-type` and `product-category` labels.
*   `Hazards_UNLABELLED_TEST.csv`: A test set containing 997 records with the same features but without the labels. The goal is to predict these missing labels.

## Methodology

The pipeline follows these key steps:

1.  **Data Preprocessing:**
    *   The `title` and `text` columns are merged into a single `full-text` field to create a comprehensive source for NLP.
    *   Categorical features (`country`, `hazard-type`, `product-category`) are numerically encoded using `LabelEncoder`.

2.  **Feature Engineering:**
    *   A simple feature, `text_len` (the length of the description), is created to capture additional signal.
    *   **TF-IDF Vectorization:** The `full-text` data is transformed into a 20,000-feature sparse matrix using `TfidfVectorizer`. This process identifies the most significant words and n-grams (1 to 3-word phrases).

3.  **Feature Combination:**
    *   The structured features (`year`, `month`, `day`, `country`, `text_len`) are combined with the TF-IDF text features into a single feature matrix for the models.

4.  **Model Training:**
    *   The training data is split into a training set (80%) and a validation set (20%).
    *   Two separate **LightGBM classifiers** are trained: one for `hazard-type` and one for `product-category`.
    *   **Class weights** are applied to handle the imbalanced nature of the target labels.
    *   **Early stopping** is used to monitor performance on the validation set and prevent overfitting, ensuring the models generalize well.

5.  **Prediction and Submission:**
    *   The trained models predict the probabilities for each class on the unseen test data.
    *   The predicted class indices are converted back to their original string labels.
    *   A final `submission.csv` file is generated containing the `ID` for each test record and its predicted `hazard` and `product`.

## Performance

The models achieved the following **Macro F1-Scores** on the local validation set, demonstrating strong predictive performance:

*   **Validation Macro F1-score (Hazard Type):** `0.7744`
*   **Validation Macro F1-score (Product Category):** `0.8252`
