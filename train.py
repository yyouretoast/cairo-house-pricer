import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def clean_sqm(value):

    if pd.isna(value):
        return None
    value = str(value)

    match = re.search(r'([\d,]+)\s*sqm', value)
    if match:
        clean_num = match.group(1).replace(',', '')
        return float(clean_num)
    return None

def clean_currency(value):

    if pd.isna(value):
        return None
    value = str(value)

    clean_num = re.sub(r'[^\d]', '', value)
    if clean_num:
        return float(clean_num)
    return None

def clean_simple_number(value):

    if pd.isna(value):
        return None
    value = str(value)
    match = re.search(r'(\d+)', value)
    if match:
        return int(match.group(1))
    return None

# load dataset
print(" Loading dataset...")
df = pd.read_csv('data/cairo-house-prices.csv') 

# clean
print("Cleaning data...")

# clean 'price'
df['price'] = df['price'].apply(clean_currency)

# clean 'size' (extracting sqm)
df['size_sqm'] = df['size'].apply(clean_sqm)

# clean 'bedrooms' and 'bathrooms'
df['bedrooms'] = df['bedrooms'].apply(clean_simple_number)
df['bathrooms'] = df['bathrooms'].apply(clean_simple_number)

# drop rows with missing critical values
df = df.dropna(subset=['price', 'size_sqm', 'bedrooms', 'location', 'type'])

# encode
print(" Encoding locations and types...")

le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location'].astype(str))

le_type = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'].astype(str))

# select features and target
X = df[['size_sqm', 'bedrooms', 'bathrooms', 'location_encoded', 'type_encoded']]
y = df['price']

print(f" Training on {len(df)} clean rows...")

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
score = model.score(X_test, y_test)
print(f" Model R^2 Score (Accuracy): {score:.2f}")

# save
print(" Saving model and encoders...")
joblib.dump(model, 'cairo_house_model.pkl', compress=3)
joblib.dump(le_location, 'location_encoder.pkl')
joblib.dump(le_type, 'type_encoder.pkl')

print("Done! Files saved: cairo_house_model.pkl, location_encoder.pkl, type_encoder.pkl")