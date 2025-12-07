import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# mock data for now
data = {
    'area': [100, 150, 200, 120, 180, 250, 80, 300],
    'bedrooms': [2, 3, 4, 2, 3, 4, 1, 5],
    'location_code': [0, 1, 2, 0, 1, 2, 0, 2], # 0: Nasr City, 1: Maadi, 2: New Cairo
    'price': [2000000, 4000000, 6000000, 2500000, 4500000, 7500000, 1500000, 9000000]
}
df = pd.DataFrame(data)

# training
X = df[['area', 'bedrooms', 'location_code']]
y = df['price']

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# save
joblib.dump(model, 'cairo_house_model.pkl')
print("Model trained and saved as cairo_house_model.pkl")