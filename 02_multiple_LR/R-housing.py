import pandas as pd
import joblib
# Loading the model and predicting the house price
housing_model = joblib.load('linear_regression_model.pkl')
input_data = pd.DataFrame({
    'area': [1500],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [1],
    'parking': [2],
    'mainroad_yes': [1],
    'guestroom_yes': [0],
    'basement_yes': [1],
    'hotwaterheating_yes': [1],
    'airconditioning_yes': [0],
    'prefarea_yes': [1],
    'furnishingstatus_semi-furnished': [1],
    'furnishingstatus_unfurnished': [0]
})
print(input_data)

# Predict the price
predicted_price = housing_model.predict(input_data)[0]
predicted_price

