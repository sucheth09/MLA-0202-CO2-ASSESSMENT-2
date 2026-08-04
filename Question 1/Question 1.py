import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Area': [500, 700, 900, 1100, 1300, 1500, 1800, 2000],
    'Rooms': [1, 2, 2, 3, 3, 4, 4, 5],
    'Location': ['City', 'Suburb', 'City', 'Suburb',
                 'City', 'Suburb', 'City', 'Suburb'],
    'Rent': [10000, 14000, 18000, 20000,
             25000, 28000, 35000, 40000]
}

df = pd.DataFrame(data)

# Convert location into numerical values
df = pd.get_dummies(df, columns=['Location'], dtype=int)

# Input and output
X = df[['Area', 'Rooms', 'Location_City', 'Location_Suburb']]
y = df['Rent']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict rent for a new apartment
new_apartment = [[1200, 3, 1, 0]]

prediction = model.predict(new_apartment)

print("Predicted Apartment Rent:", prediction[0])
