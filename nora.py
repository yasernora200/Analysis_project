import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from pandas.api.types import is_numeric_dtype
import warnings

warnings.filterwarnings("ignore")

# 1. تحميل البيانات
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# 2. تحديد نوع المهمة
def detect_task_type(df, target_column):
    if is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 10:
        return "regression"
    return "classification"

# 3. تجهيز البيانات
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = pd.get_dummies(X)
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()


# 4. الحصول على الموديلات المناسبة
def get_models(task):
    if task == "classification":
        return {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier()
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "SVR": SVR()
        }

# 5. الحصول على المقاييس المتاحة
def get_metrics(task):
    if task == "classification":
        return {
            "Accuracy": accuracy_score,
            "F1 Score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score
        }
    else:
        return {
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
            "R2": r2_score
        }

# 6. تدريب موديل معين
def train_single_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# 7. تقييم موديل معين
def evaluate_model(model, X_test, y_test, metric_func):
    y_pred = model.predict(X_test)
    if "average" in metric_func.__code__.co_varnames:
        return metric_func(y_test, y_pred, average="macro")
    return metric_func(y_test, y_pred)

# 8. حفظ الموديل والخصائص
def save_model(model, feature_columns, filename="best_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

# 9. تحميل موديل محفوظ
def load_model(filename="best_model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

# 10. التنبؤ ببيانات جديدة
def predict_new_data(model, feature_columns, new_df):
    new_data = pd.get_dummies(new_df)
    for col in feature_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[feature_columns]
    return model.predict(new_data)
