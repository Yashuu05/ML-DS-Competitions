import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import mlflow
import optuna

# 1. Move data loading outside the objective function to avoid redundant I/O operations
print("Loading data...")
train_df = pd.read_csv(r"Data\train.csv")
test_df = pd.read_csv(r"Data\test.csv")

train_df = train_df.drop("id", axis=1)

cat_cols = []
num_cols = []

# Exclude class from features
features_df = train_df.drop("class", axis=1)
for cols in features_df.columns:
    if features_df[cols].dtypes == "object":
        cat_cols.append(cols)
    else:
        num_cols.append(cols)
        
# prepare input and output
y = train_df["class"]
X = train_df.drop("class", axis=1)

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

def objective(trial):
    # 2. Corrected Hyperparameters for RandomForest
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    # create a pipeline
    cat_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy="most_frequent")),
        ('encode', OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    num_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    preprocssor = ColumnTransformer(transformers=[
        ('num', num_preprocessor, num_cols),
        ('cat', cat_preprocessor, cat_cols)
    ])
    
    model = Pipeline(steps=[
        ('prep', preprocssor),
        ('model', RandomForestClassifier(
            **params,
            random_state=0,
            n_jobs=-1            
        ))
    ])
    
    with mlflow.start_run(nested=True):
        print("Training model...")
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of model: {acc}")
    
        mlflow.log_metric("accuracy", acc)

        # 3. Return accuracy for Optuna to optimize
        return acc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name="Steller Classification Competition")
mlflow.sklearn.autolog()

with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_accuracy", study.best_value)

    # 4. Generate final submission only using the best model
    print("Training final model with best parameters...")
    
    best_params = study.best_params
    
    cat_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy="most_frequent")),
        ('encode', OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    num_preprocessor = Pipeline(steps=[
        ('impute', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    preprocssor = ColumnTransformer(transformers=[
        ('num', num_preprocessor, num_cols),
        ('cat', cat_preprocessor, cat_cols)
    ])
    
    best_model = Pipeline(steps=[
        ('prep', preprocssor),
        ('model', RandomForestClassifier(
            **best_params,
            random_state=0,
            n_jobs=-1            
        ))
    ])
    
    # Train best model on full data
    best_model.fit(X, y) 
    
    print("Generating submission file...")
    ids = test_df['id']
    test_features = test_df.drop("id", axis=1)
    test_preds = best_model.predict(test_features)
    
    test_preds_decoded = le.inverse_transform(test_preds)
    
    os.makedirs("Results", exist_ok=True)
    submission = pd.DataFrame({
        "id": ids,
        "class": test_preds_decoded
    })
    submission.to_csv("Results/submission3.csv", index=False)
    print("Submission saved to Results/submission3.csv")