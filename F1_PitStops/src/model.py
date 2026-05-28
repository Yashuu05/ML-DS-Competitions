from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

models = {
    "RandomForest":RandomForestClassifier(),
    "LinearRegression":LinearRegression(),
    "LightGBM":LGBMClassifier(),
    "XGBoost":XGBClassifier()
}