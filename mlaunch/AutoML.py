from typing import Literal

from mlaunch.core.data_io import load_dataset
from mlaunch.core.data_info import dataset_info
from mlaunch.core.data_cleaning import data_cleaning
from mlaunch.core.preprocessing import preprocessing
from mlaunch.core.modeling import creating_model,choose_model

def AutoML(path:str,y_column:str,model_name:Literal["Linear Regression","Logistic Regression","Random Forest Regression","Hist Gradient Boosting Regression","Random Forest classifier","Hist Gradient Boosting classifier","Auto Select"] = "Auto Select",type:Literal["python file","pipeline"] = "pipeline"):
    """
    Preprocesses the dataset and builds a machine learning model.

    Parameters
    ----------
    path : str
        Path to the CSV dataset.
    y_column : str
        Name of the target column in the dataset.
    model_name : Literal[
        "Linear Regression",
        "Logistic Regression",
        "Random Forest Regression",
        "Hist Gradient Boosting Regression",
        "Random Forest Classifier",
        "Hist Gradient Boosting Classifier",
        "Auto Select"
    ]
        Model to use.  
        - Auto Select chooses the best model based on the data.  
        - Additional models may be added in the future.
    type : Literal["pipeline", "python file"]
        Output format:
        - "pipeline": returns a trained sklearn Pipeline.
        - "python file": exports the model to `model.pkl` and runs a generated Python file for inference.

    Returns
    -------
    Pipeline | tuple
        - If type="pipeline": returns the trained Pipeline.
        - If type="python file": returns the output artifacts (e.g., file path, score).
    """

    df =load_dataset(path)
    print("cleaning data...")
    df = data_cleaning(df,y_column)
    model = choose_model(model_name,df,y_column)
    print("preprocessing...")
    preprocessor = preprocessing(model,df,y_column)
    print("creating model...")
    folder_path,score = creating_model(model,df,y_column,preprocessor,path,type="python file")
    return folder_path,score