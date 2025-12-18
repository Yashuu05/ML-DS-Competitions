# 🚀 Spaceship Titanic – Machine Learning Classification

This repository contains my solution to the **Kaggle Spaceship Titanic** competition.  
The goal of the competition is to predict whether a passenger was **transported to an alternate dimension** after the Spaceship Titanic collided with a spacetime anomaly.

The project focuses on **data preprocessing, feature engineering, model training, and evaluation** using multiple machine learning algorithms.

---

## 📌 Problem Statement

The Spaceship Titanic was carrying nearly 13,000 passengers when it encountered a spacetime anomaly during its maiden voyage. As a result, almost half of the passengers were transported to another dimension.

Using the recovered passenger records, the task is to build a machine learning model that predicts:

> **Was the passenger transported? (True / False)**

This is a **binary classification problem**.

---

## 📊 Dataset Description

The dataset is provided by Kaggle and includes passenger-level information such as:

- `PassengerId`
- `HomePlanet`
- `CryoSleep`
- `Cabin`
- `Destination`
- `Age`
- `VIP`
- Spending details:
  - `RoomService`
  - `FoodCourt`
  - `ShoppingMall`
  - `Spa`
  - `VRDeck`
- `Name`
- `Transported` (Target Variable)

---

## 🧠 Approach & Methodology

The notebook follows a structured machine learning workflow:

### 1. Data Loading & Exploration
- Load training and test datasets
- Understand feature distributions and missing values

### 2. Feature Engineering
- Split `Cabin` into:
  - `Deck`
  - `CabinNum`
  - `Side`
- Drop non-informative columns (`PassengerId`, `Name`)
- Handle missing values appropriately

### 3. Preprocessing
- **Numerical Features**
  - Standard Scaling
- **Categorical Features**
  - One-Hot Encoding
- Implemented using `ColumnTransformer`

### 4. Model Training
The following models were trained and evaluated using **scikit-learn Pipelines**:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **LightGBM Classifier**

### 5. Hyperparameter Tuning
- `GridSearchCV` for Random Forest
- `RandomizedSearchCV` for XGBoost and LightGBM
- 5-fold cross-validation
- Evaluation metric: **Accuracy**

### 6. Model Evaluation
- Compare cross-validation scores
- Evaluate performance on validation/test data

---

## 🧪 Models Used

| Model              | Library        |
|-------------------|----------------|
| Random Forest      | scikit-learn  |
| XGBoost Classifier | xgboost       |
| LightGBM Classifier| lightgbm      |

---

## Results

- Multiple models were trained and tuned using cross-validation.
- Tree-based ensemble models performed best.
- Hyperparameter tuning improved generalization performance.
- Final predictions were generated in Kaggle submission format.

---

## Kaggle Competition

- Competition: Spaceship Titanic
- Type: Binary Classification
- Metric: Accuracy, r2 score

**Competition Link: https://www.kaggle.com/competitions/spaceship-titanic**

---

## Future Improvements

- Feature importance analysis
- SHAP-based model explainability
- Model stacking / ensembling
- Automated missing value imputation strategies

---

## Acknowledgements

- Kaggle for providing the dataset and competition
- scikit-learn, XGBoost, and LightGBM communities

---

If you have any suggestions or feedback, feel free to connect or open an issue.

Happy Learning! 🚀

---