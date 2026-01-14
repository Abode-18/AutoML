import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings

# ignore all warnings
warnings.filterwarnings("ignore")

def get_data():
    global path
    path = input("Enter the path to the dataset: ")
    df = pd.DataFrame(pd.read_csv(path))
    print("choose the y column of these columns\n")
    for column in df.columns.tolist():
        print(column)
    global y_column
    y_column = input("choose the y column in the dataframe: ")
    print("Dataset loaded successfully.")
    return df
def get_info_about_df(df):
    global cat_columns,num_columns,dataset_size
    dataset_size = len(df)
    cat_columns = df.select_dtypes(include = object).columns.tolist()
    num_columns = df.select_dtypes(include = ['int64','float64']).columns.tolist()
    for column in df.columns:
        if column in cat_columns:
            continue
        elif column in num_columns:
            if df[column].nunique() < 10:
                del num_columns[num_columns.index(column)]
def get_info_about_column(df):
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

def split(df):
    X_train,X_test,y_train,y_test = train_test_split(df.drop(y_column,axis = 1),df[y_column],test_size=0.3,random_state=1)
    return X_train,X_test,y_train,y_test




def main_auto():
    pass

def choosing_model():
    models_names = ["Logistic Regression","Random Forest Regression","Hist Gradient Boosting Regression","Random Forest classifier","Hist Gradient Boosting classifier","autoselect - it will take a while"]
    print("Choose a model from the following options:")
    for modelName in models_names:
        print(f"{models_names.index(modelName)+1} ",modelName)
    choose = int(input("Enter the number corresponding to your choice: "))
    global model_name
    if choose == 1:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model_name = "Logistic Regression"
    elif choose == 2:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model_name = "Random Forest Regression"
    elif choose == 3:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor()
        model_name = "Hist Gradient Boosting Regression"
    elif choose == 4:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model_name = "Random Forest classifier"
    elif choose == 5:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier()
        model_name = "Hist Gradient Boosting classifier"
    else:
        pass
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.ensemble import RandomForestRegressor
        # from sklearn.ensemble import HistGradientBoostingRegressor
        # from sklearn.ensemble import RandomForestClassifier
        # from sklearn.ensemble import HistGradientBoostingClassifier
        # from sklearn.model_selection import train_test_split
        # models = [LogisticRegression(),RandomForestRegressor(),HistGradientBoostingRegressor(),RandomForestClassifier(),HistGradientBoostingClassifier()]
        # Reg_models = [LogisticRegression(),RandomForestRegressor(),HistGradientBoostingRegressor()]
        # Clas_models = [RandomForestClassifier(),HistGradientBoostingClassifier()]
        

        # global X_train,X_test,y_test,y_pred,y_train
        # X_train,X_test,y_train,y_test = train_test_split(df.drop(y_column,axis = 1),df[y_column],test_size=0.3,random_state=1)
        # biggest_score = Reg_model_evaluation(Reg_models[0])
        # for test_model in models:
        #     if test_model in Reg_models:
        #         if Reg_model_evaluation(test_model) > biggest_score:
        #             biggest_score = Reg_model_evaluation(test_model) 
        #             model = test_model
        #     elif test_model in Clas_models:
        #         if Class_model_evaluation(test_model) > biggest_score:
        #             biggest_score = Class_model_evaluation(test_model)
        #             model = test_model
    return model




def get_model_type(model):
    from sklearn.base import RegressorMixin, ClassifierMixin
    if isinstance(model, RegressorMixin):
        return "regression"
    elif isinstance(model, ClassifierMixin):
        return "classification"
        

            




def delete_outliers(df,column):
    if dataset_size >=100000 and info[column]["outliers_ratio"]>= 0.02:
        lower = df[column].quantile(0.01)
        upper = df[column].quantile(0.99)
        df[column] = df[column].clip(lower = lower,upper = upper)
    elif info[column]["outliers_ratio"] <= 0.01 and info[column]["max_zscore"] >= 5:
        from scipy.stats import zscore
        col = df[column]  # column you want to clean
        z = abs(zscore(col))
        outliers = z > 5
        df = df[~outliers] # keep only non-outlier rows
    elif (info[column]["outliers_ratio"] > 0.01 and info[column]["outliers_ratio"] <= 0.05) or (info[column]["max_zscore"]>=3 and info[column]["max_zscore"]<5) or (np.abs(info[column]["skewness"])>1):
        col = df[column]
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Cap values instead of deleting
        df[column] = col.clip(lower=lower, upper=upper)
    elif info[column]["outliers_ratio"] > 0.05 and np.abs(info[column]["skewness"])>=2:
        if (df[column]>0).all():
            df[column] = np.log1p(df[column])
        else:
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df[column] = pt.fit_transform(df[[column]]).flatten()
    elif np.abs(info[column]["skewness"])<0.5 and (info[column]["kurtosis"]>=2.5 and info[column]["kurtosis"]<=3.5):
        z = abs(zscore(df[column]))
        mask = z<=3
        df = df[mask]
    return df



def choosing_outlier_method(model,df,df_outliers):
    from sklearn.preprocessing import QuantileTransformer,RobustScaler
    X_train,X_test,y_train,y_test = split(df_outliers)
    model.fit(X_train,y_train)
    if model_type == "regression":
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        score = r2_score(y_test,y_pred)
    elif model_type == "classification":
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test,y_pred)
    X_train,X_test,y_train,y_test = split(df)
    pipe = Pipeline([
        ("scaler",QuantileTransformer()),
        ("model",model)
    ])
    mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[QuantileTransformer(),RobustScaler()]})
    mod.fit(X_train,y_train)
    grid_score = mod.best_score_
    if grid_score > score:
        return mod.best_params_["scaler"]
    else:
        return None

def choose_scaler(df):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    X_train,X_test,y_train,y_test = split(df)
    pipe = Pipeline([
        ("scaler",StandardScaler()),
        ("model",model)
    ])
    mod = GridSearchCV(estimator=pipe,param_grid={"scaler":[StandardScaler(),MinMaxScaler()]})
    mod.fit(X_train,y_train)
    return mod.best_params_["scaler"]


def encoding(df):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    X_train,X_test,y_train,y_test = split(df)
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns),
        ('num', 'passthrough', num_columns)
    ])
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ]
    )
    param_grid = {
        'preprocessor__cat': [
            OneHotEncoder(handle_unknown='ignore'),
            OrdinalEncoder()
        ]
    }
    grid = GridSearchCV(pipe, param_grid, cv=3)
    grid.fit(X_train,y_train)
    encoder = grid.best_params_['preprocessor__cat']
    if isinstance(encoder, OrdinalEncoder):
        for column in cat_columns:
            df[column] = encoder.fit_transform(df[[column]])
    else:
        for column in cat_columns:
            encoded = encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(encoded.toarray() if hasattr(encoded, "toarray") else encoded,
                                  columns=encoder.get_feature_names_out([column]),
                                  index=df.index)
            df = df.drop(column, axis=1).join(encoded_df)
    return df


def creating_model(model,scaler,df):
    import joblib
    import os
    X_train,X_test,y_train,y_test = split(df)
    model = Pipeline([
        ("scaler",scaler),
        ("model",model)
    ])
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    if model_type == "regression":
        from sklearn.metrics import r2_score
        score = r2_score(y_test,y_pred)
        print(score)
    elif model_type == "classification":
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test,y_pred)
        print(score)
    folder_path = os.path.join(os.path.dirname(path),"my_ML_model")
    os.makedirs(folder_path, exist_ok=True)
    joblib.dump(model,os.path.join(folder_path,"model.pkl"))
    df.to_csv(os.path.join(folder_path,os.path.basename(path)))
    print("model created succesfully🥳\n","model: ",folder_path)
    
    
    

def main():
    global info,model,model_type
    df=get_data()
    get_info_about_df(df)
    info = get_info_about_column(df)
    model = choosing_model()
    df = encoding(df)
    get_info_about_df(df)
    model_type = get_model_type(model)
    for column in num_columns:
        df_outliers = delete_outliers(df,column)
    if choosing_outlier_method(model,df,df_outliers) == None:
        df = df_outliers
        scaler = choose_scaler(df)
    else:
        scaler = choosing_outlier_method(model,df,df_outliers)
    info = get_info_about_column(df)
    creating_model(model,scaler,df)
    

main()
