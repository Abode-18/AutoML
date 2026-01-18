import os
from typing import Literal

import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import GridSearchCV
import joblib

from mlaunch.core.utils import split
from mlaunch.core.preprocessing import preprocessing

def get_model_type(model):
    # Unwrap GridSearchCV or Pipeline
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]  # last step is the estimator

    if isinstance(model, RegressorMixin):
        return "regression"
    elif isinstance(model, ClassifierMixin):
        return "classification"
    else:
        raise ValueError("Unknown model type")

def creating_model(model,df:pd.DataFrame,y_column,preprocessor,path,type: Literal['pipeline', 'python file']):
    X_train,X_test,y_train,y_test = split(df,y_column)

    if isinstance(model, LinearRegression):
        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",LinearRegression())
        ])
        
        model = GridSearchCV(estimator=pipe,param_grid={"model__fit_intercept":[True,False],"model__positive":[True,False]})
    elif isinstance(model, LogisticRegression):
        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",LogisticRegression())
        ])
        
        param_grid = {
            "model__penalty": ['l1', 'l2', 'elasticnet', 'none'],
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ['liblinear', 'saga'],
            "model__max_iter": [100, 200, 500],
            "model__class_weight": [None, 'balanced']
        }

        model = GridSearchCV(estimator=pipe,param_grid=param_grid)


    elif isinstance(model, RandomForestRegressor):
        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",RandomForestRegressor())
        ])
        param_grid = {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ['auto', 'sqrt', 'log2', 0.5],
            "model__bootstrap": [True, False]
        }

        model = GridSearchCV(estimator=pipe,param_grid=param_grid)


    elif isinstance(model, RandomForestClassifier):

        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",RandomForestClassifier())
        ])

        param_grid = {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ['auto', 'sqrt', 'log2', 0.5],
            "model__bootstrap": [True, False],
            "model__class_weight": [None, 'balanced', 'balanced_subsample'],
            "model__criterion": ['gini', 'entropy', 'log_loss']
        }

        model = GridSearchCV(estimator=pipe,param_grid=param_grid)


    elif isinstance(model, HistGradientBoostingRegressor):
        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",HistGradientBoostingRegressor())
        ])
        param_grid = {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_iter": [100, 200, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [20, 50, 100],
            "model__max_bins": [255, 512],
            "model__l2_regularization": [0.0, 0.1, 0.5],
            "model__loss": ['squared_error', 'absolute_error']
        }

        model = GridSearchCV(estimator=pipe,param_grid=param_grid)


    elif isinstance(model, HistGradientBoostingClassifier):
        pipe = Pipeline([
            ("preprocessor",preprocessor),
            ("model",HistGradientBoostingClassifier())
        ])

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [100, 200, 500],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [20, 50, 100],
            "max_bins": [255, 512],
            "l2_regularization": [0.0, 0.1, 0.5],
            "loss": ['log_loss'],
            "max_leaf_nodes": [31, 63, 127]
        }
        model = GridSearchCV(estimator=pipe,param_grid=param_grid)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    if get_model_type(model) == "regression":
        from sklearn.metrics import r2_score
        score = r2_score(y_test,y_pred)
    elif get_model_type(model) == "classification":
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test,y_pred)
    
    if type == "python file":
        folder_path = os.path.join(os.path.dirname(path),"my_ML_model")
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(model,os.path.join(folder_path,"model.pkl"))
        df.to_csv(os.path.join(folder_path,os.path.basename(path)),index=False)
        with open(os.path.join(folder_path,"ML_model.py"),"w") as f:
            f.write(
f"""
import joblib
import pandas as pd

# Load the model
model = joblib.load("{os.path.join(folder_path,"model.pkl").replace("\\", "/")}")
data = pd.read_csv("{os.path.join(folder_path,os.path.basename(path)).replace("\\", "/")}")
# Make predictions
new_data = []
X_columns = data.drop("{y_column}", axis=1).columns.tolist()
for column in X_columns:
    if pd.api.types.is_integer_dtype(data[column]):
        new_data.append(int(input(f"enter value for {{column}}: ")))
    elif pd.api.types.is_float_dtype(data[column]):
        new_data.append(float(input(f"enter value for {{column}}: ")))
    else:
        new_data.append(input(f"enter value for {{column}}: "))
new_data = pd.DataFrame([new_data], columns=X_columns)
predictions = model.predict(new_data)  # new_data should match the format of your CSV

print("{y_column}: ",predictions)
""")

        return folder_path,score
    elif type == "pipeline":
        return model

           

def auto_select_model(df,y_column):
    preprocessor = preprocessing(LinearRegression(),df,y_column)
    X_train,X_test,y_train,y_test = split(df,y_column)
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("model", LinearRegression())
    ])

    grid = GridSearchCV(estimator=pipe,param_grid={"model":[LinearRegression(),LogisticRegression(),RandomForestRegressor(),HistGradientBoostingRegressor(),RandomForestClassifier(),HistGradientBoostingClassifier()]})

    grid.fit(X_train,y_train)
    return grid.best_params_["model"]


def choose_model(name: str, df,  y_column:list):
    models = {
        "Linear Regression": LinearRegression,
        "Logistic Regression": LogisticRegression,
        "Random Forest Regression": RandomForestRegressor,
        "Hist Gradient Boosting Regression": HistGradientBoostingRegressor,
        "Random Forest classifier": RandomForestClassifier,
        "Hist Gradient Boosting classifier": HistGradientBoostingClassifier,
        "Auto Select": auto_select_model
    }
    if name == "Auto Select":
        return models[name](df, y_column)
    else:
        return models[name]() 

