# AutoML

## About the Project
AutoML Builder is a Python project that automatically generates a complete machine learning model from a CSV file. It handles data preprocessing, encoding, model selection, and evaluation with minimal user input.

## Features
- Automatic detection of feature types (numerical, categorical, ordinal)
- Handles missing values and outliers
- Selects and trains the best ML model
- Provides evaluation metrics for regression and classification
- Easy integration with custom pipelines

## Installation
```bash
pip install automl
```
## How to use
1. create a python file and paste this code:
```python
import pandas as pd
import warnings
import os
import subprocess

from automl import AutoML

warnings.filterwarnings("ignore")


path = input("enter the path for the dataset: ")
df = pd.read_csv(path)
for column in df.columns.tolist():
    print(column)
y_column = input("choose the y column in the dataframe: ")
models_names = ["Linear Regression","Logistic Regression","Random Forest Regression","Hist Gradient Boosting Regression","Random Forest classifier","Hist Gradient Boosting classifier","Auto Select"]
for model_name in models_names:
    print(f"{models_names.index(model_name) + 1}- ",model_name)
model_name = models_names[int(input("choose a model by writing the number crossponding with the model you want: "))-1]

folder_path,score = AutoML(path,y_column,model_name)
print("model craeted sucessfully 🥳")
print(f"path: {folder_path}")
print(f"score: {score}")
subprocess.run(["python", os.path.join(folder_path,"ML_model.py")], check=True)
```
# package
## Functions
### AutoML
this function will preprocess the data and create the model for you
```python
from automl import AutoML
model = AutoML(path,y_column,model_name,type = "pipeline")
```
#### parameteres
- path: the path for your csv file
- y_column: the target column in your dataset
- model_name: the name of your model out of these current model:
    - Linear Regression
    - Logistic Regression
    - Random Forest Regression
    - Hist Gradient Boosting Regression
    - Random Forest classifier
    - Hist Gradient Boosting classifier
    - Auto Select : it will select the best model for your data
    more will be added in the future
- type: how will the output be:
    - pipeline: it will output the model as a pipeline
    - python file: it will export the model to `model.pkl` file and run a python file to input the data

### preprocessing
this function will handle the outliers and encode your data and output it as a ColumnTransformer
```python 
from automl import preprocessing
preprocessor = preprocessing(model,df,y_column)
model = Pipeline([
    ("preprocessor",preprocessor)
])
```
#### parameteres
- model: put your model here
- df: put the dataframe here
- y_column: the target column in your dataset

### dataset_info
this function returns a dictionary of the size, cat_columns and num_columns
```python
from automl import dataset_info
info = dataset_info(df,y_column)
```
#### parameteres
- df: put the dataframe here
- y_column: the target column in your dataset

### column_statistics
this function returns a bunch of stats for every column in the dataframe
```python
from automl import column_statistics
stats = column_statistics(df,cat_columns,num_columns)
```
#### parameteres
- df: put the dataframe here
- cat_columns: the catagorical columns in your dataframe, you can get them easily by using `dataset_info(df,y_column)["cat_columns]`
- num_columns: the numerical columns in your dataframe, you can get them easily by using `dataset_info(df,y_column)["num_columns]`