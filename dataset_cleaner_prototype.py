import numpy as np
import pandas as pd

def get_data():
    path = input("Enter the path to the dataset: ")
    df = pd.DataFrame(pd.read_csv(path))
    df['Oldpeak'].value_counts()
    print("Dataset loaded successfully.")
    return df
def get_info_about_df(df):
    global cat_columns,num_columns
    cat_columns = df.select_dtypes(include = object).columns.tolist()
    num_columns = df.select_dtypes(include = int).columns.tolist()
    for column in df.columns:
        if column in cat_columns:
            continue
        elif column in num_columns:
            if df[column].value_counts() < 10:
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
def Reg_model_evaluation(model):
        from sklearn.metrics import r2_score
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test,y_pred)
        return score
def Class_model_evaluation(model):
    pass

def choosing_model():

    models = ["Logistic Regression","Random Forest Regression","Hist Gradient Boosting Regression","Random Forest classifier","Hist Gradient Boosting classifier","autoselect - it will take a while"]
    # models = {
    #     "Regression":{
    #         "Logistic Regression": LogisticRegression(),
    #         "Random Forest": RandomForestRegressor(),
    #         "Hist Gradient Boosting": HistGradientBoostingRegressor()
    #     },
    #     "Classification":{
    #         "Random Forest": RandomForestClassifier(),
    #         "Hist Gradient Boosting": HistGradientBoostingClassifier()
    #     }
    #     }
    print("Choose a model from the following options:")
    for model in models:
        print(f"{models.index(model)+1} ",model)
    choose = int(input("Enter the number corresponding to your choice: "))

    if choose == 1:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif choose == 2:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor
    elif choose == 3:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor
    elif choose == 4:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif choose == 5:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier()
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        models = [LogisticRegression(),RandomForestRegressor(),HistGradientBoostingRegressor(),RandomForestClassifier(),HistGradientBoostingClassifier()]
        Reg_models = [LogisticRegression(),RandomForestRegressor(),HistGradientBoostingRegressor()]
        Clas_models = [RandomForestClassifier(),HistGradientBoostingClassifier()]
        
        print("choose the y column of these columns\n")
        for column in df.columns.tolist():
            print(column)
        y_column = input("choose the y column in the dataframe: ")
        global X_train,X_test,y_test,y_pred,y_train
        X_train,X_test,y_train,y_test = train_test_split(df.drop(y_column,axis = 1),df[y_column],test_size=0.3,random_state=1)
        biggest_score = Reg_model_evaluation(Reg_models[0])
        for test_model in models:
            if test_model in Reg_models:
                if Reg_model_evaluation(test_model) > biggest_score:
                    biggest_score = Reg_model_evaluation(test_model) 
                    model = test_model
            elif test_model in Clas_models:

        

            




def delete_outliers(df):
    num_col = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_col:
        pass



def main():
    global df
    df=get_data()
    choosing_model()

main()
