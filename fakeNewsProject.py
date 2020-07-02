import numpy as np
import pandas as pd
import sklearn
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading the data
df=pd.read_csv('news.csv')  
df.shape
print(df.head())

#Get the labels
labels=df.label
labels.head()

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DFit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Prediction
y_pred=pac.predict(tfidf_test)
print(y_test)
print(y_pred)

#calculate accuracy
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Confusion Matrix
cf = confusion_matrix(y_test, y_pred)
print(cf)
#Extracting individual values from the confusion matrix
TN = cf[0,0]
FP = cf[0,1]
FN = cf[1,0]
TP = cf[1,1]
print("True Positive value:",TP)
print("False Positive value:",FP)
print("True Negative value:",TN)
print("False Negative value:",FN)

# Bar Graph for Real news
rlabel = [ 'Correctly Predicted','Incorrectly Predicted']
rvalues=[TP,FN]
plt.bar(rlabel,rvalues,width=0.5,color = ['r','k'])
plt.xlabel('FN vs TP')
plt.ylabel('Predicted values')
plt.title('Bar Graph : Real News Prediction')
plt.show()

# Bar Graph for Fake news
flabel = [ 'Correctly Predicted','Incorrectly Predicted']
fvalues=[TN,FP]
plt.bar(flabel,fvalues,width=0.5,color = ['r','k'])
plt.xlabel('TN vs FP')
plt.ylabel('Predicted values')
plt.title('Bar Graph : Fake News Prediction')
plt.show()
