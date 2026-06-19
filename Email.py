import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('emails.csv')
print(data.head())
print(data.dtypes)
print(data.describe())
print(data.isnull().sum())

x = data['text']
y = data['label']

# print(x)
# print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(data['text'])

from sklearn.model_selection import train_test_split
x_train , x_test ,  y_train , y_test = train_test_split(x , y , test_size= 0.20 , random_state = 42)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy Score :" , accuracy)


while True:
    user = input("Please Enter The Email :")
    if user == "q":
        break
    new_email_user = tfidf.transform([user])
    result = mnb.predict(new_email_user)
    print(le.inverse_transform(result))