# AutoML Builder

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
git clone https://github.com/Abode-18/AutoML.git
cd AutoML
pip install -r requirements.txt
```
## How to use
1. Run `main.py`, then provide the path to your CSV file.  
The program will create a `.pkl` file inside a folder named `my_model` located in the same directory as your CSV file.  

2. To use the saved model, create a Python file in the my_model folder and add the following code (replace new_data with your input data):
```python
import joblib

# Load the model
model = joblib.load("path/to/my_ML_model/model.pkl")

# Make predictions
predictions = model.predict(new_data)  # new_data should match the format of your CSV

print(predictions)
