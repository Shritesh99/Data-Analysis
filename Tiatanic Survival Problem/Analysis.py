
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
train = pd.read_csv('dataset/train.csv')
train = train.dropna(subset=['Age','Embarked','Fare'])
test = pd.read_csv('dataset/test.csv')
test_result = pd.read_csv('dataset/gender_submission.csv')
test = test.join(test_result['Survived'])
test = test.dropna(subset=['Age','Embarked','Fare'])

# Selecting desireable fields
x_train = train.iloc[:, [2, 4, 5, 6, 7, 9, 10, 11]].values
y_train = train.iloc[:, 1].values

x_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10]].values
y_test = test.iloc[:, 11].values

# Data Preprocessing

# Taking care of missing data
for x in range(len(x_train)):
    if str(x_train[x, 6]) == "nan": 
        x_train[x, 6] = 0
    else : 
        x_train[x, 6] = len(str(x_train[x, 6]).split(" "))
      
for x in range(len(x_test)):
    if str(x_test[x, 6]) == "nan": 
        x_test[x, 6] = 0
    else : 
        x_test[x, 6] = len(str(x_test[x, 6]).split(" "))
    
# Taking care of Categorical Data
for x in range(len(x_train)):
    if str(x_train[x, 1]) == "male": 
        x_train[x, 1] = 1
    else : 
        x_train[x, 1] = 0     
        
for x in range(len(x_test)):
    if str(x_test[x, 1]) == "male": 
        x_test[x, 1] = 1
    else : 
        x_test[x, 1] = 0
        
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
x_train[:, 7] = labelencoder_X.fit_transform(x_train[:, 7])

labelencoder_y = LabelEncoder()
x_test[:, 7] = labelencoder_y.fit_transform(x_test[:, 7])        

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # accuracy = 80%

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # accuracy = 80%

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # accuracy = 79%

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # accuracy = 91%

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # accuracy = 80%

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # accuracy = 95%

# SVM with linear kernal gives best accuracy.
        