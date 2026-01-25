import numpy as np
import pandas as pd

from mlaunch.core.data_info import dataset_info,column_statistics

def handle_missing(df:pd.DataFrame,y_column):
    custom_missing = [    
    "", " ", "  ",
    "nan", "NaN", "NAN",
    "null", "NULL", "Null",
    "none", "None", "NONE",
    "nil", "NIL",
    "na", "NA", "N/A", "n/a",
    "missing", "MISSING",
    "-", "--", "---",
    None,
    "?"
    ]
    df.replace(custom_missing, pd.NA, inplace=True)
    df = df.dropna(subset=[y_column],axis=0)

    for column in dataset_info(df,y_column)["num_columns"]:
        if column_statistics(df,y_column)[column]["missing_ratio"] < 0.05:
            column_missing = df[column].isna()
            grouped = df.groupby(column_missing)[column].mean()
            diff = grouped.diff().abs()

            if len(diff) > 1:
                mean_diff = diff.iloc[1]
            else:
                mean_diff = 0

            if mean_diff < 0.05 :
                df = df.dropna(subset=[column])
            else:
                df[column] = df[column].fillna(df[column].median())
        elif column_statistics(df,y_column)[column]["missing_ratio"] > 0.1 and column_statistics(df,y_column)[column]["missing_ratio"] < 0.2:
            if column_statistics(df,y_column)[column]["outliers_ratio"] < 0.05:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].median())
        elif column_statistics(df,y_column)[column]["missing_ratio"] > 0.2:
            df = df.drop(columns=[column],axis=1)
    return df

def id_columns_deleting(df:pd.DataFrame,y_column) -> pd.DataFrame:
    for column in dataset_info(df,y_column)["cat_columns"]:
        if df[column].nunique() /dataset_info(df,y_column)["size"] >= 0.90:
            df = df.drop(column,axis=1)
    return df

def feature_deleting(df:pd.DataFrame,y_column:str) -> pd.DataFrame:
    cat_columns = dataset_info(df,y_column)["cat_columns"]
    num_columns = dataset_info(df,y_column)["num_columns"]
    df = handle_missing(df,y_column)
    df = id_columns_deleting(df,y_column)
    return df

def remove_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df
def data_cleaning(df:pd.DataFrame,y_column:str) -> pd.DataFrame:
    """
    this function will handle NaN and remove duplicates
    
    :param df: your DataFrame
    :type df: pd.DataFrame
    :param y_column: the target column in your dataset
    :type y_column: str
    :return: the DataFrame after cleaning
    :rtype: DataFrame
    """
    df = feature_deleting(df,y_column)
    df = remove_duplicates(df)
    return df