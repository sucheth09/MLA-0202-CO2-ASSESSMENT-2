import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    'Free': [5, 4, 0, 1, 6, 0, 1, 5],
    'Money': [4, 3, 0, 1, 5, 0, 0, 4],
    'Offer': [3, 4, 0, 1, 5, 0, 1, 3],
    'Spam': [1, 1, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Input features
X = df[['Free', 'Money', 'Offer']]

# Target
y = df['Spam']

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# New email word frequencies
new_email = [[4, 3, 2]]

# Prediction
prediction = model.predict(new_email)

if prediction[0] == 1:
    print("Email is SPAM")
else:
    print("Email is NOT SPAM")
