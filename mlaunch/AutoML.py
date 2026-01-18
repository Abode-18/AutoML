from typing import Literal

from mlaunch.core.data_io import load_dataset
from mlaunch.core.preprocessing import preprocessing
from mlaunch.core.modeling import creating_model,choose_model

def AutoML(path:str,y_column:str,model_name:Literal["Linear Regression","Logistic Regression","Random Forest Regression","Hist Gradient Boosting Regression","Random Forest classifier","Hist Gradient Boosting classifier","Auto Select"],type:Literal["python file","pipeline"] = "pipeline"):

    df =load_dataset(path)
    model = choose_model(model_name,df,y_column)
    preprocessor = preprocessing(model,df,y_column)
    folder_path,score = creating_model(model,df,y_column,preprocessor,path,type="python file")
    return folder_path,score