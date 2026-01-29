# *Richter’s Predictor: Modeling Earthquake Damage*

---

- Platform: DrivenData
- Competition: Richter’s Predictor – Modeling Earthquake Damage
- URL : https://www.drivendata.org/competitions/57/nepal-earthquake.
- Level: Intermediate (Practice Competition)
- Current Rank = *1720 / 2634*
- Public Score = *0.7215*

- Based on aspects of building location and construction, the goal of this competition is to predict the level of damage sustained by buildings during the 2015 Gorkha earthquake in Nepal.

The target variable damage_grade is an ordinal multiclass label:

`1` – Low damage
`2` – Medium damage
`3` – Almost complete destruction

The challenge focuses on applying real-world machine learning techniques to disaster-impact data, balancing predictive performance with responsible modeling decisions

---

## Dataset Description

- The dataset was collected through large-scale post-disaster surveys conducted by:
> Kathmandu Living Labs
> Central Bureau of Statistics (CBS), Nepal (under the National Planning Commission Secretariat)

This is one of the largest post-disaster datasets ever collected, containing rich information about:
- Structural characteristics of buildings
- Geographic location (multi-level administrative regions)
- Construction materials and design
- Usage patterns and occupancy

---

## Problem Type

*Multiclass classification (ordinal)*

Class distribution is moderately imbalanced, with damage grade 2 being the majority class
Because the labels are discrete and ordinal, the problem is treated as classification, not regression.

---

## Files Used

- `train_values.csv` – Input features for training
- `train_labels.csv` – Target labels (damage_grade)
- `test_values.csv` – Test features for final prediction
- The dataset contains `39` columns, including:
    - `building_id` (unique identifier)
    - 38 feature columns (numerical + categorical)

---

## Approach & Methodology

1. *Exploratory Analysis*

- Studied class imbalance in damage_grade
- Identified numerical and categorical features
- Observed strong dependence on construction material, age, and location

2. *Data Preprocessing*

- Used ColumnTransformer for clean preprocessing
- Applied OneHotEncoding to categorical features
- Passed numerical features without scaling (tree-based models)

All preprocessing steps were embedded inside a scikit-learn Pipeline to Prevent data leakage and to ensure consistency between training and test data.

---

## Models Implemented

1. Baseline Model
    - RandomForestClassifier
    - Tuned using GridSearchCV
    - Added class_weight='balanced' to handle imbalance

2. Advanced Models
    A. XGBoost (XGBClassifier)
    - Required converting class labels from {1,2,3} to {0,1,2}
    - Used multi:softprob objective

    B. LightGBM (LGBMClassifier) 
    - Faster training
    - Better handling of categorical-heavy data
    - Native support for class weighting

Hyperparameters were tuned using _cross-validation_, primarily optimizing _F1-micro_, while final performance was judged using _F1 score (micro)_, as per DrivenData’s evaluation metric.

--- 

## Evaluation Metrics
- Accuracy 
- F1-score (micro) (primary competition metric)
- Classification Report to ensure minority class learning

⚠️ Regression metrics such as R² score were intentionally avoided, as the task is classification-based.

---

## Submission Process

- Predictions generated on `test_values.csv`

- Submission file format:
`building_id`,`damage_grade`

---

## Key Learnings

- Tree-based boosting models outperform bagging models on structured tabular data
- Proper metric selection is crucial (classification vs regression)
- Class imbalance does not always require resampling
- Pipelines are essential for safe and reproducible ML workflows
- XGBoost enforces zero-based class indexing for multiclass problems

---

## Future Improvements

- Model ensembling (LightGBM + XGBoost)
- Feature importance and explainability analysis
- Ordinal classification–specific loss functions
- Probability calibration and confidence-based predictions

---

## Acknowledgements

- DrivenData for hosting socially impactful machine learning competitions
- Kathmandu Living Labs & Central Bureau of Statistics, Nepal for making this valuable dataset available.
- Kaggle for providing Free computational resources.
- The open-source ML community for continuous learning resources.