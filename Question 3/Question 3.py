import pandas as pd
from sklearn.naive_bayes import GaussianNB


data = {
    'Fever': [1, 1, 0, 0, 1, 0, 1, 0],
    'Cough': [1, 1, 1, 0, 1, 0, 1, 0],
    'Headache': [1, 0, 1, 0, 1, 0, 1, 0],
    'Flu': [1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)


X = df[['Fever', 'Cough', 'Headache']]


y = df['Flu']


model = GaussianNB()


model.fit(X, y)


patient = [[1, 1, 1]]


prediction = model.predict(patient)

if prediction[0] == 1:
    print("Patient is likely to have FLU")
else:
    print("Patient is NOT likely to have FLU")
