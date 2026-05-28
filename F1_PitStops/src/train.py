import os 
import sys 
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = sys.path.insert(0, root)
from src import utils
from src.hyperparameters import RandomForest, LogisticRegression, LightGBM, XGBoost
from src.model import models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# step 1
print("step 1: loading dataset")
try: 
    train_df = pd.read_csv(f"{os.path.join(PROJECT_ROOT, "dataset", "train.csv")}")
    test_df = pd.read_csv(f"{os.path.join(PROJECT_ROOT, "dataset", "test.csv")}")

except Exception as e:
    print(f"Error! {e}")

# step 2
print("step 2 : Applying feature engineering")
new_train_df = utils.feature_engineering(df=train_df)
new_test_df = utils.feature_engineering(df=test_df)

# step 3 
print("step 3 : preparing labels and features")
y = new_train_df["PitNextLap"]
X = new_train_df.drop(["id", "PitNextLap"], axis=1)
test_id = new_test_df["id"]
test_df = test_df.drop("id", axis=1)
print("X: \n", X.columns)
print("\ny: \n", y.columns)

# step 4
print("step 4: Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=True, shuffle=True
    )
print(f"\nx train: {X_train.shape} | y train: {y_train.shape}")
print(f"\nx test: {X_test.shape} | y test : {y_test.shape}")

# step 5
print("step 5 : creating list of categorical and numerical features")
cat_cols, num_cols = utils.find_cat_num_cols(df=X_train)

# step 6
print("step 6: creating pipeline")
pipeline = utils.create_pipeline(categorical_col=cat_cols, numerical_col=num_cols)
rf_model = models["RandomForest"]
rf_pipeline = Pipeline(steps=[
    ("prep", pipeline),
    ("model", rf_model)
])
rf_grid = utils.grid_search_cv(model_pipeline=rf_pipeline, param_grid=RandomForest, cv=3)

# step 7
print("step 7: training model")
rf_grid.fit(X_train, y_train)
print("best parameters = ", rf_grid.best_params_)
print("best score = ", rf_grid.best_score_)

# step 8
print("step 8: evaluating model")
roc = utils.evaluate_model(model=rf_grid.best_estimator_, X_test=X_test, y_test=y_test)
print("ROC SCORE = ", roc)

# step 9
print("step 9: make predictions")
predictions = utils.make_predictions(model=rf_grid.best_estimator_, df=new_test_df)

# step 10
print("step 10: prepare the submission file")
utils.prepare_output(name="submission_3", test_id=test_id, test_predictions=predictions)

