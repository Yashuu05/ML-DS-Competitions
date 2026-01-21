# *Abalone Ring Prediction – Kaggle Competition*

- *Competition page* : https://www.kaggle.com/competitions/testingtechno

---

## Competition Overview

- This project is part of a Kaggle regression competition where the objective is to predict the age of abalone using its physical measurements.
- Instead of manually counting shell rings (a time-consuming and error-prone process), machine learning models are used to estimate the number of rings based on measurable attributes such as length, diameter, height, and weight.
- The competition evaluates predictions using Root Mean Squared Logarithmic Error (RMSLE), which emphasizes relative prediction accuracy and penalizes large underestimations more strongly.

---

## Problem Statement

- Input: Physical measurements of abalone
- Output: Number of shell rings (Rings)
- Learning Type: Supervised Machine Learning (Regression)
- Evaluation Metric: RMSLE
- `Note: Abalone age is approximately calculated as Age = Rings + 1.5 years.`

---

## Dataset Description

- The dataset consists of two main files:

- `train.csv` – Contains features along with the target variable (Rings)
- `test.csv` – Contains only features; Rings must be predicted

*Key Features:*
- `Sex` – Categorical feature `(M, F, I)`
- `Length`, `Diameter`, `Height` – Physical dimensions
- `Whole weight`, `Shucked weight`, `Viscera weight`, `Shell weight` – Weight-based features
- `Rings` – Target variable (only in training data)
`Note : Dataset is not shared considering Competition Rules.`

---

## My Work & Approach

All experimentation and analysis are implemented in testTechno.ipynb.
The workflow followed in this notebook is outlined below:

### 1. Data Loading & Inspection

- Loaded training data using Pandas
- Checked dataset shape, data types, and descriptive statistics
- Verified missing values and duplicate records

### 2. Data Cleaning & Feature Engineering

- Removed non-informative columns such as id
- Renamed ambiguous column names (e.g., Whole weight.1 → Shucked weight) for clarity
- Ensured consistent numerical formatting across features

### 3. Exploratory Data Analysis (EDA)

- Visualized feature distributions using histograms
- Analyzed feature–target relationships using scatter plots
- Identified skewness and potential outliers in weight-related features

### 4. Outlier Handling

- Addressed skewed numerical features using log transformation
- Preserved biological extremes instead of removing valid rare observations
- Ensured transformations aligned with the RMSLE evaluation metric

### 5. Feature Scaling

- Applied StandardScaler to numerical features
- Scaling was used to stabilize learning and improve model convergence

### 6. Model Training

- Trained baseline and improved models using:
    1. Random Forest Regressor
- Tuned parameters such as:
`n_estimators`, `max_depth`, `min_samples_split`

### 7. Model Evaluation & Submission

- Generated multiple submission files for comparison
- Ensured correct submission format (id, Rings)
- Exported predictions as CSV files ready for Kaggle upload

### 8. Results & Observations

- Log transformation significantly improved model stability
- Tree-based models handled feature distributions effectively
- Avoiding aggressive outlier removal helped preserve predictive patterns
- Incremental preprocessing led to noticeable performance improvements

---

## Tools & Technologies Used

1. Python
2. Pandas, NumPy
3. Matplotlib, Seaborn
4. Scikit-learn
5. Google Colab Notebbok
6. Kaggle Platform

---

## Appreciation

- I would like to thank Kaggle for providing an accessible platform to practice real-world machine learning problems and Google for Free Colab Notebook.
- This competition helped me strengthen my understanding of:
    1. Regression modeling
    2. Feature preprocessing
    3. Outlier handling
    4. RMSLE-oriented optimization

- It was a valuable hands-on learning experience.

---

## Future Improvements

- Experiment with XGBoost / LightGBM
- Perform cross-validation for robust RMSLE estimation
- Try target (Rings) log transformation with inverse prediction
- Hyperparameter tuning using GridSearch or Bayesian Optimization

---