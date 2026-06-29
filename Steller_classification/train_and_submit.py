import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import mlflow
from mlflow.models import infer_signature

def main():
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
            
    y = train_df["class"]
    X = train_df.drop("class", axis=1)
    
    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    signature = infer_signature(X_train, y_train)

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
        ('model', DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=0))
    ])
    
    with mlflow.start_run():

        print("Training model...")
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of model: {acc}")
    
        mlflow.log_metric("accuracy", acc)
        #mlflow.sklearn.log_model(model, "model", signature=signature)

        print("Generating submission file...")
        ids = test_df['id']
        test_features = test_df.drop("id", axis=1)
        test_preds = model.predict(test_features)
    
        test_preds_decoded = le.inverse_transform(test_preds)
    
        os.makedirs("Results", exist_ok=True)
        submission = pd.DataFrame({
            "id": ids,
            "class": test_preds_decoded
        })
        submission.to_csv("Results/submission1.csv", index=False)
        print("Submission saved to Results/submission1.csv")

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name="Steller Classification Competition")
    mlflow.autolog()
    main()
