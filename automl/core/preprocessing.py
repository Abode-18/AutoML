import numpy as np

import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer
)
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from automl.core.utils import split
from automl.core.evaluation import evaluate
from automl.core.data_info import dataset_info,column_statistics

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        self.lower_ = X.quantile(self.lower_q)
        self.upper_ = X.quantile(self.upper_q)
        return self

    def transform(self, X):
        return X.clip(self.lower_, self.upper_, axis=1)

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        self.lower_ = Q1 - self.factor * (Q3 - Q1)
        self.upper_ = Q3 + self.factor * (Q3 - Q1)
        return self

    def transform(self, X):
        return X.clip(self.lower_, self.upper_, axis=1)
    
class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.positive_only_ = (X > 0).all().all()
        if not self.positive_only_:
            self.pt_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.pt_.fit(X)
        return self

    def transform(self, X):
        if self.positive_only_:
            return np.log1p(X)
        return self.pt_.transform(X)




def choose_scaler(df,model,column,y_column):
    pipe = Pipeline([
        ("scaler",StandardScaler()),
        ("model",model)
    ])
    mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[StandardScaler(),MinMaxScaler(),RobustScaler(),MaxAbsScaler(),PowerTransformer(method='yeo-johnson'),QuantileTransformer(output_distribution='normal')]})
    mod.fit(df[[column]],df[y_column])
    best_scaler_class = type(mod.best_params_["scaler"])
    if isinstance(mod.best_params_["scaler"], PowerTransformer):
        return PowerTransformer(method='yeo-johnson')
    elif isinstance(mod.best_params_["scaler"], QuantileTransformer):
        return QuantileTransformer(output_distribution='normal')
    else:
        return best_scaler_class()

def handling_outliers(df,model,num_columns,stats,y_column,transformers:list):
    dataset_size = dataset_info(df,y_column)["size"]
    # df_outliers = df.copy()
    QuantileClipper_columns = []
    IQRClipper_columns = []
    SkewnessTransformer_columns = []
    for column in num_columns:
        if dataset_size >=100000 and stats[column]["outliers_ratio"]>= 0.02:
            QuantileClipper_columns.append(column)
        # elif stats[column]["outliers_ratio"] <= 0.01 and stats[column]["max_zscore"] >= 5:
        #     from scipy.stats import zscore
        #     col = df_outliers[column]  # column you want to clean
        #     z = abs(zscore(col))
        #     outliers = z > 5
        #     df_outliers = df_outliers[~outliers] # keep only non-outlier rows
        elif (stats[column]["outliers_ratio"] > 0.01 and stats[column]["outliers_ratio"] <= 0.05) or (stats[column]["max_zscore"]>=3 and stats[column]["max_zscore"]<5) or (np.abs(stats[column]["skewness"])>1):
            IQRClipper_columns.append(column)
        elif stats[column]["outliers_ratio"] > 0.05 and np.abs(stats[column]["skewness"])>=2:
            SkewnessTransformer_columns.append(column)
        # elif np.abs(stats[column]["skewness"])<0.5 and (stats[column]["kurtosis"]>=2.5 and stats[column]["kurtosis"]<=3.5):
        #     z = abs(zscore(df_outliers[column]))
        #     mask = z<=3
        #     df_outliers = df_outliers[mask]
        # else:
        #     transformers.append((f"{column} scaler",choose_scaler(df,model,column,y_column),column))
        transformers.append(("QuantileClipper",QuantileClipper(),QuantileClipper_columns))
        transformers.append(("IQRClipper",IQRClipper(),IQRClipper_columns))
        transformers.append(("SkewnessTransformer",SkewnessTransformer(),SkewnessTransformer_columns))
        return transformers
    ####
    # X_train,X_test,y_train,y_test = split(df_outliers,y_column)

    # pipe = Pipeline([
    #     ("preprocessor",preprocessor),
    #     ("model", model)
    # ])

    # score = evaluate(pipe,X_train,X_test,y_train,y_test)

    # X_train,X_test,y_train,y_test = split(df,y_column)
    # pipe = Pipeline([
    #     ("preprocessor",preprocessor),
    #     ("scaler",QuantileTransformer()),
    #     ("model",model)
    # ])
    # mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[QuantileTransformer(),RobustScaler()]})
    # mod.fit(X_train,y_train)
    # grid_score = mod.best_score_
    # if grid_score > score:
    #     return df,scaler
               
    # else:
    #     df = df_outliers
    #     scaler = choose_scaler(df,model,y_column,preprocessor)
    #     return df,scaler
               
    
def OHE_score(model,df:pd.DataFrame,column,y_column,OHE_columns,OE_columns,TE_columns):
    ohe_cols = OHE_columns + [column]
    X_train, X_test, y_train, y_test = split(df, y_column)
    transformers = []
    if ohe_cols:
        transformers.append(("OHE", OneHotEncoder(handle_unknown="ignore"), ohe_cols))
    if OE_columns:
        transformers.append(("OE", OrdinalEncoder(), OE_columns))
    if TE_columns:
        for col in TE_columns:
            transformers.append(("TE_" + col, TargetEncoder(), [col])) 
    preprocessor = ColumnTransformer(transformers, remainder=OneHotEncoder(handle_unknown="ignore"))

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return evaluate(pipe, X_train, X_test, y_train, y_test)

def OE_score(model,df:pd.DataFrame,column,y_column,OHE_columns,OE_columns,TE_columns):
    oe_cols = OE_columns + [column]
    X_train, X_test, y_train, y_test = split(df, y_column)
    transformers = []
    if OHE_columns:
        transformers.append(("OHE", OneHotEncoder(handle_unknown="ignore"), OHE_columns))
    if oe_cols:
        transformers.append(("OE", OrdinalEncoder(), oe_cols))
    if TE_columns:
        for col in TE_columns:
            transformers.append(("TE_" + col, TargetEncoder(), [col])) 
    preprocessor = ColumnTransformer(transformers, remainder=OneHotEncoder(handle_unknown="ignore"))

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return evaluate(pipe, X_train, X_test, y_train, y_test)

def TE_score(model,df:pd.DataFrame,column,y_column,OHE_columns,OE_columns,TE_columns):
    te_cols = TE_columns + [column]
    X_train, X_test, y_train, y_test = split(df, y_column)
    transformers = []
    if OHE_columns:
        transformers.append(("OHE", OneHotEncoder(handle_unknown="ignore"), OHE_columns))
    if OE_columns:
        transformers.append(("OE", OrdinalEncoder(), OE_columns))
    if te_cols:
        for col in te_cols:
            transformers.append(("TE_" + col, TargetEncoder(), [col])) 
    preprocessor = ColumnTransformer(transformers, remainder=OneHotEncoder(handle_unknown="ignore"))

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return evaluate(pipe, X_train, X_test, y_train, y_test)



def choose_encoders(df:pd.DataFrame,model,cat_columns,y_column):
    df_test = df.copy()
    transformers = []
    OHE_columns = []
    OE_columns = []
    TE_columns = []
    for column in cat_columns:
        if OHE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) > OE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) and OHE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) > TE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns):
            OHE_columns.append(column)
        elif OE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) > OHE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) and OE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns) > TE_score(model,df_test,column,y_column,OHE_columns,OE_columns,TE_columns):
            OE_columns.append(column)
        else:
            TE_columns.append(column)
    transformers.append(("OHE", OneHotEncoder(handle_unknown="ignore"), OHE_columns))
    transformers.append(("OE", OrdinalEncoder(), OE_columns))
    transformers.append(("TE", TargetEncoder(), TE_columns))


    return transformers

def preprocessing(model,df:pd.DataFrame,y_column):
    cat_columns = dataset_info(df,y_column)["cat_columns"]
    num_columns = dataset_info(df,y_column)["num_columns"]
    stats = column_statistics(df,cat_columns,num_columns)
    transformers = choose_encoders(df,model,cat_columns,y_column)
    transformers = handling_outliers(df,model,num_columns,stats,y_column,transformers)
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor