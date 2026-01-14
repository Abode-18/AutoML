from sklearn.model_selection import train_test_split
import pandas as pd
def split(df:pd.DataFrame,y_column):
    X_train,X_test,y_train,y_test = train_test_split(df.drop(y_column,axis = 1),df[y_column],test_size=0.3,random_state=1)
    return X_train,X_test,y_train,y_test