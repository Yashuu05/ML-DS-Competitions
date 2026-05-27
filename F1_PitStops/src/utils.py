from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import os 
import sys 
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = sys.path.insert(0, root)

def feature_engineering(df):
    """
    Applies feature engineering to original dataset
    Input : raw dataset
    Output : new dataset
    """
    df = df.copy()

    df["lap_per_tyrelife"] = (df["LapNumber"] / (df["TyreLife"] + 1))
    df["degradation_per_tyrelife"] = (df["Cumulative_Degradation"] / df["TyreLife"])
    df["agressive_climb"] = (df["Position"] / (df["LapNumber"] + 1))

    return df

def find_cat_num_cols(df) -> tuple:
    """
    Creates the list of categorical and numnerical column names of provided dataset
    Input : dataset / dataframe
    Output : list categorical and numnerical columns 
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    
    print("num columns = ", num_cols)
    print("\n cat columns = ", cat_cols)

    return cat_cols, num_cols

def create_pipeline(categorical_col: list, numerical_col: list):
    """
    creates the pipeline by integrating OneHotEncoding, StandardScaler and SimpleImputer
    for categrical and numerical features

    Input: categorical and numnerical column list
    categorical_col = list of "object" features
    numerical_col = list of int/float features

    Output : Pipeline
    """
    categorical_preprocessor = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_preprocessor, categorical_col),
        ("num", numerical_preprocessor, numerical_col)
    ])

    return preprocessor

def prepare_output(name: str, test_id, test_predictions):
    """
    creates a required Submission csv file having "id" and predicted probability of "NextPitLap" and
    saves in outputs/ folder.

    Input:
    name = name of the output file
    test_id = "id" of test dataset
    test_predictions = predicated probabilities of "NextPitLap"  

    Output:
    csv file with "id" and "PitNextLap" column
    """

    submission = pd.DataFrame({
        "id" : test_id,
        "PitNextLap" : test_predictions
    })
    os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, "outputs")), exist_ok=True)
    submission.to_csv(f"{os.path.join(PROJECT_ROOT, "outputs", {name.csv})}", index=False)
    print(submission.head())
    
    print(f"\n{submission.shape}")
    print(f"\nis null \n= {submission.isnull().sum()}")
    print(f"\n{name} saved successfully")


def make_predictions(model, df):
    """
    returns the array of probabilities of Target Feature

    Inputs:
    model = trained or saved model on trian dataset
    df = test dataset having same features as train dataset
    """
    test_predictions = model.predict_proba(df)[:, 1]
    return test_predictions

def evaluate_model(model, X_test, y_test):
    """
    returns the "roc_auc_score" and "classification report" of the trained model on the basis of test features
    
    Inputs:
    model = trained / saved model on train dataset
    X_test = test dataset features
    y_test = test dataset target labels

    Output:
    roc_auc_score
    
    Note: According to the requirements of competition, roc_auc_score was selected as primary
    metrics to measure performance of the model.  
    """
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    print("roc score = ", roc)
    print("classification report = \n", classification_report(y_test, y_pred))

    return roc

def stratified_k_fold(random_state=42, n_splits=5):
    """
    Performs training in specified number of folds of training dataset

    Input:
    random_state: default = 42, int
    n_splits: default = 5, int

    output: stratified k fold
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    return skf

def grid_search_cv(model_pipeline, param_grid: dict, cv=5):
    """
    applies the hyperparameter tuning to the model to determine best hyperparameters for given model

    Input:
    model_pipeline : pre-built model pipeline
    param_grid : dictionary of model's hyperparameter
    cv : number of cross validation, default = 5

    Output:
    Grid Search CV
    """
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        n_jobs=-1
    )
    
    return grid_search