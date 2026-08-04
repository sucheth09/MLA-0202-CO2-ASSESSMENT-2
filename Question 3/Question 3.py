import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Dataset
data = {
    'Fever': [1, 1, 0, 0, 1, 0, 1, 0],
    'Cough': [1, 1, 1, 0, 1, 0, 1, 0],
    'Headache': [1, 0, 1, 0, 1, 0, 1, 0],
    'Flu': [1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Input features
X = df[['Fever', 'Cough', 'Headache']]

# Target
y = df['Flu']

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X, y)

# New patient's symptoms
# Fever = 1, Cough = 1, Headache = 1
patient = [[1, 1, 1]]

# Prediction
prediction = model.predict(patient)

if prediction[0] == 1:
    print("Patient is likely to have FLU")
else:
    print("Patient is NOT likely to have FLU")
