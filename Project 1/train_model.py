# run in Treminal --> python train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data from CSV
df = pd.read_csv(r'D:\Class Laptop\Class Data analysis\Machine Learning\CSV fILE\Dummy_Data.csv')

# 2. Manual Preprocessing
# Extract Date Features
if 'Built_Date' in df.columns:
    df['Built_Date'] = df['Built_Date'].astype('datetime64[ns]')
    df['Built_Year'] = df['Built_Date'].dt.year
    df['Built_Month'] = df['Built_Date'].dt.month

# Mathematical Combination
df['Area_per_Bedroom'] = df['Area_sqft'] / df['Bedrooms']

# Define Features (X) and Target (y)
X = df.drop(['Price', 'Built_Date'], axis=1, errors='ignore')
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define Preprocessing for specific columns
numeric_features = ['Area_sqft', 'Bedrooms', 'Property_Tax', 'Built_Year', 'Built_Month', 'Area_per_Bedroom']
categorical_features = ['Neighborhood']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 4. Build the Simplified Pipeline (Feature Selection Removed)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 5. Train and Evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("--- Pipeline Evaluation ---")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# 6. Predict on New Raw Data (Testing multiple Area_sqft values to see the price change)
new_data = pd.DataFrame({
    'Area_sqft': [1400, 2500, 100000], # Testing 3 different sizes
    'Bedrooms': [3, 3, 3],
    'Neighborhood': ['Suburb', 'Suburb', 'Suburb'],
    'Built_Date': ['2026-10-26', '2026-10-26', '2026-10-26'],
    'Property_Tax': [np.nan, np.nan, np.nan]
})

# Manually preprocess the new data before feeding it to the pipeline
new_data['Built_Date'] = pd.to_datetime(new_data['Built_Date'])
new_data['Built_Year'] = new_data['Built_Date'].dt.year
new_data['Built_Month'] = new_data['Built_Date'].dt.month
new_data = new_data.drop('Built_Date', axis=1)
new_data['Area_per_Bedroom'] = new_data['Area_sqft'] / new_data['Bedrooms']

print("\n--- Pipeline Prediction ---")
predictions = pipeline.predict(new_data)

# Print the changing predictions
for area, price in zip(new_data['Area_sqft'], predictions):
    print(f"Predicted Price for {area} sqft: ${price:,.2f}")
    
    
# 6. Save the Model
joblib.dump(pipeline, 'property_price_model.joblib')
print("Model saved successfully as 'property_price_model.joblib'")