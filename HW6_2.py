from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import warnings
import numpy

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data = pd.read_csv('/Users/sana/Downloads/Data/Practical 2/wifi_data.txt', delim_whitespace=True, header=None ,names= ["X_1", "X_2", "X_3", "X_4","X_5","X_6","X_7","Y"])
count = 0

for i in data.values:
    if i[7] == 2 or i[7] == 4:
        # print(i,count)
        data = data.drop(count)
        # count = count - 1
    count = count + 1
data = data.sample(frac=1)
X =  data[["X_1", "X_2", "X_3", "X_4","X_5","X_6","X_7"]].to_numpy()
Y =  data[["Y"]].to_numpy()
# print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
lr = LogisticRegression(multi_class= 'auto')
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over 1 and 3 with zero req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 0.5)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over 1 and 3 with 0.5 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over 1 and 3 with 1 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 2)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over 1 and 3 with 2 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))


data2 = pd.read_csv('/Users/sana/Downloads/Data/Practical 2/wifi_data.txt', delim_whitespace=True, header=None ,names= ["X_1", "X_2", "X_3", "X_4","X_5","X_6","X_7","Y"])
data2 = data2.sample(frac=1)
X =  data2[["X_1", "X_2", "X_3", "X_4","X_5","X_6","X_7"]].to_numpy()
Y =  data2[["Y"]].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
lr = LogisticRegression(multi_class= 'auto')
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over all with zero req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 0.5)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over all with 0.5 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over all with 1 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))
lr = LogisticRegression(multi_class= 'auto',penalty='l2', C = 2)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("over all with 2 req:")
print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy",accuracy_score(y_test,y_pred))

