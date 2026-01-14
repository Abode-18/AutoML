import numpy as np

import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from core.utils import split
from core.evaluation import evaluate
from core.data_info import dataset_info


def choose_scaler(df,cat_columns,model,y_column,encoder):
    X_train,X_test,y_train,y_test = split(df,y_column)
    preprocessor = ColumnTransformer([
        ("cat",encoder,cat_columns)
    ])
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("scaler",StandardScaler()),
        ("model",model)
    ])
    mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[StandardScaler(),MinMaxScaler()]})
    mod.fit(X_train,y_train)
    return mod.best_params_["scaler"]

def handling_outliers(model,df,num_columns,cat_columns,stats,y_column,encoder):
    dataset_size = dataset_info(df)["size"]
    df_outliers = df.copy()
    for column in num_columns:
        if dataset_size >=100000 and stats[column]["outliers_ratio"]>= 0.02:
            lower = df_outliers[column].quantile(0.01)
            upper = df_outliers[column].quantile(0.99)
            df_outliers[column] = df_outliers[column].clip(lower = lower,upper = upper)
        elif stats[column]["outliers_ratio"] <= 0.01 and stats[column]["max_zscore"] >= 5:
            from scipy.stats import zscore
            col = df_outliers[column]  # column you want to clean
            z = abs(zscore(col))
            outliers = z > 5
            df_outliers = df_outliers[~outliers] # keep only non-outlier rows
        elif (stats[column]["outliers_ratio"] > 0.01 and stats[column]["outliers_ratio"] <= 0.05) or (stats[column]["max_zscore"]>=3 and stats[column]["max_zscore"]<5) or (np.abs(stats[column]["skewness"])>1):
            col = df_outliers[column]
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # Cap values instead of deleting
            df_outliers[column] = col.clip(lower=lower, upper=upper)
        elif stats[column]["outliers_ratio"] > 0.05 and np.abs(stats[column]["skewness"])>=2:
            if (df_outliers[column]>0).all():
                df_outliers[column] = np.log1p(df_outliers[column])
            else:
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                df_outliers[column] = pt.fit_transform(df_outliers[[column]]).flatten()
        elif np.abs(stats[column]["skewness"])<0.5 and (stats[column]["kurtosis"]>=2.5 and stats[column]["kurtosis"]<=3.5):
            z = abs(zscore(df_outliers[column]))
            mask = z<=3
            df_outliers = df_outliers[mask]
    ####
    X_train,X_test,y_train,y_test = split(df_outliers,y_column)
    preprocessor = ColumnTransformer([
        ("cat",encoder,cat_columns)
    ])

    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("model", model)
    ])

    score = evaluate(pipe,X_train,X_test,y_train,y_test)

    X_train,X_test,y_train,y_test = split(df,y_column)
    preprocessor = ColumnTransformer([
        ("cat",encoder,cat_columns)
    ])
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("scaler",QuantileTransformer()),
        ("model",model)
    ])
    mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[QuantileTransformer(),RobustScaler()]})
    mod.fit(X_train,y_train)
    grid_score = mod.best_score_
    if grid_score > score:
        return df,scaler
               
    else:
        df = df_outliers
        scaler = choose_scaler(df,cat_columns,model,y_column,encoder)
        return df,scaler
               
    


def choose_encoder(df:pd.DataFrame,model,cat_columns,y_column):
    X_train,X_test,y_train,y_test = split(df,y_column)
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
    ])
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ]
    )
    param_grid = {
        'preprocessor': [
            ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
            ]),
            ColumnTransformer([
                ('cat', OrdinalEncoder(), cat_columns)
            ])
        ]
    }

    grid = GridSearchCV(pipe, param_grid, cv=3)
    grid.fit(X_train,y_train)
    encoder = grid.best_estimator_.named_steps['preprocessor'].named_transformers_['cat']
    return encoder