import numpy as np

import pandas as pd



def dataset_info(df:pd.DataFrame,y_column:str) -> dict:
    """this function returns a dictionary of the size, cat_columns and num_columns"""
    dataset_size = len(df)
    cat_columns = df.drop(y_column,axis=1).select_dtypes(include = object).columns.tolist()
    num_columns = df.drop(y_column,axis=1).select_dtypes(include = ['int64','float64']).columns.tolist()
    for column in df.columns:
        if column in cat_columns:
            continue
        elif column in num_columns:
            if df[column].nunique() < 10:
                del num_columns[num_columns.index(column)]
    return {
        "size":dataset_size,
        "cat_columns" : cat_columns,
        "num_columns" : num_columns
    }

def column_statistics(df:pd.DataFrame,cat_columns:list[str],num_columns:list[str]) ->dict:
    """this function returns a bunch of stats for every column in the dataframe"""
    stats = {}
    for column in df.columns:
        info = {}
        info["missing_ratio"] = df[column].isnull().sum() / len(df[column])
        if column in cat_columns:
            info['n_unique'] = df[column].nunique()
        if column in num_columns:
            info["mean"] = df[column].mean()
            info["std"] = df[column].std()
            info["min"] = df[column].min()
            info["max"] = df[column].max()
            info["skewness"] = df[column].skew()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            info["outliers_ratio"] = len(outliers) / len(df)
            info["kurtosis"] = df[column].kurt()
            info["max_zscore"] = ((df[column] - df[column].mean())/df[column].std()).max()
            stats[column] = info
    return stats

