
import joblib
import pandas as pd

# Load the model
model = joblib.load("tests/my_ML_model/model.pkl")
data = pd.read_csv("tests/my_ML_model/phone.csv")
# Make predictions
new_data = []
X_columns = data.drop("price_range", axis=1).columns.tolist()
for column in X_columns:
    if pd.api.types.is_integer_dtype(data[column]):
        new_data.append(int(input(f"enter value for {column}: ")))
    elif pd.api.types.is_float_dtype(data[column]):
        new_data.append(float(input(f"enter value for {column}: ")))
    else:
        new_data.append(input(f"enter value for {column}: "))
new_data = pd.DataFrame([new_data], columns=X_columns)
predictions = model.predict(new_data)  # new_data should match the format of your CSV

print("price_range: ",predictions)
