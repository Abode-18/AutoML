import pandas as pd
import warnings
import os
import subprocess

from AutoML import AutoML

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
