import pandas as pd
from sklearn.linear_model import LogisticRegression


data = {
    'Free': [5, 4, 0, 1, 6, 0, 1, 5],
    'Money': [4, 3, 0, 1, 5, 0, 0, 4],
    'Offer': [3, 4, 0, 1, 5, 0, 1, 3],
    'Spam': [1, 1, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)


X = df[['Free', 'Money', 'Offer']]


y = df['Spam']


model = LogisticRegression()
model.fit(X, y)


new_email = [[4, 3, 2]]


prediction = model.predict(new_email)

if prediction[0] == 1:
    print("Email is SPAM")
else:
    print("Email is NOT SPAM")
