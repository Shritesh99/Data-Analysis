import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

train = pd.read_csv('dataset/train_V2.csv')
test = pd.read_csv('dataset/test_V2.csv')
test_result = pd.read_csv('dataset/sample_submission_V2.csv')

train = train.dropna()
train['winPlacePerc'] = train['winPlacePerc'].astype(np.int64)
test = test.join(test_result['winPlacePerc'])
test = test.dropna()

x_train = train.iloc[:, 3:28].values
y_train = train.iloc[:, -1].values

x_test = test.iloc[:, 3:28].values
y_test = test.iloc[:, -1].values

labelencoder_X_train = LabelEncoder()
labelencoder_X_test = LabelEncoder()
x_train[:, 12] = labelencoder_X_train.fit_transform(x_train[:, 12])
x_test[:, 12] = labelencoder_X_test.fit_transform(x_test[:, 12])


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("{:0.2f} % Accuracy".format(((cm[0,0]+cm[1,1])/y_test.shape[0])*100))