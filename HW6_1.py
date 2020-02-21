import random

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing, linear_model


def preprocess(df):
    df['gender'].replace('female', 1, inplace=True)
    df['gender'].replace('male', 0, inplace=True)
    df['smoker'].replace('yes', 1, inplace=True)
    df['smoker'].replace('no', 0, inplace=True)
    df = pd.concat([df,pd.get_dummies(df['region'], prefix='region')],axis=1)
    df.drop(['region'],axis=1, inplace=True)
    df['age'] = df['age']**2
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]
    Y = df[["charges"]].to_numpy()
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    # X =  df[["region_northeast", "region_northwest","region_southwest","region_southwest","age","gender", "bmi", "children","smoker"]].to_numpy()
    X = df[[0, 1, 2, 3, 4, 5, 6, 7, 8]].to_numpy()
    # Y = df[[9]].to_numpy()
    Y = Y.flatten()
    return X,Y


def find_error(y_p, Y_test):
    e = 0
    for i in range(len(Y_test)):
        e = e + ((Y_test[i] - y_p[i]) ** 2)
    return e/len(Y_test)

def find_SSE_error(y_p, Y_test,l,w):
    e = 0
    for i in range(len(Y_test)):
        e = e + ((Y_test[i] - y_p[i]) ** 2)/2
    return e+ l * np.dot(w, w)

def BatchGradientDescent(theta, alpha, num_iters, X, Y,X_test,Y_test, n):
    ys = X.dot(theta)
    error = np.ones(num_iters)
    error_test = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/len(X)) * sum(ys - Y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/len(X))*sum((ys - Y) * X.transpose()[j])
        ys = X.dot(theta)
        y_t = X_test.dot(theta)
        error[i] = (find_error(ys,Y))
        error_test[i] = (find_error(y_t,Y_test)/len(X))
    return theta, error,error_test


# def stochastic_gradiant_descent(theta, alpha, num_iters, X, Y, n):
#     # print(X)
#     # print(Y)
#     error = []
#     m = len(Y)
#     for i in range(num_iters):
#         e = 0
#         for j in range(10):
#             # print(m)
#             r = np.random.randint(0,m)
#             # print(X[r,:])
#             X_p = X[r,:]
#             Y_p = Y[r]
#             predictions = np.dot(X_p,theta)
#             theta = theta - (1/m)*alpha*(X_p.T.dot((predictions - Y_p)))
#             print(theta)
#             predictions = np.dot(X,theta)
#             e += (1/2*m)*np.sum(np.square(predictions-Y_p))
#         error.append(e)
#     return theta,error

def getW_from_formula(X,Y):
    X_T = np.transpose(X)
    H = X_T.dot(X)
    H_I = np.linalg.pinv(H)
    J = H_I.dot(X_T)
    W = J.dot(Y)
    return W

def getW_from_formula_with_landa(X,Y,landa):
    X_T = np.transpose(X)
    H = X_T.dot(X)
    I = np.identity(len(H))
    H_I = np.linalg.pinv(H + landa*I)
    J = H_I.dot(X_T)
    W = J.dot(Y)
    return W

def five_fold_cross_validation(X, Y, landa):
    sse = 0.0
    ws = [0] * 10
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        w = getW_from_formula_with_landa(x_train,y_train, landa)
        ws = [ws[j] + w[j] for j in range(10)]
        sse += find_SSE_error(x_test.dot(w),y_test, landa,w)
    return [0.2 * s for s in ws], sse / 5

df = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/train.csv')
X,Y = preprocess(df)
# print(X)
# print(Y)
df_test = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/test.csv')
X_test, Y_test =  preprocess(df_test)

print("Part one one:")
reg = LinearRegression().fit(X,Y)
print("Coef:",reg.coef_, reg.intercept_)
y_t =  reg.predict(X)
y_p = reg.predict(X_test)
print("test error:",find_error(y_p,Y_test))
print("train error:",find_error(y_t,Y))
E_rms_train = []
E_rms_test = []
X_list = []
for i in range(100):
    reg = LinearRegression().fit(X[0:10*(i+1)], Y[0:10*(i+1)])
    y_t = reg.predict(X[0:10*(i+1)])
    y_p = reg.predict(X_test)
    # print("test error for",i*10,":",find_error(y_p, Y_test))
    # print("train error for",i*10,":",find_error(y_t, Y[0:10*(i+1)]))
    E_rms_train.append(math.sqrt(2*find_error(y_t, Y[0:10*(i+1)])))
    E_rms_test.append(math.sqrt(2 * find_error(y_p, Y_test)))
    X_list.append(10*(i+1))
plt.plot(X_list, E_rms_test)
plt.plot(X_list, E_rms_train)
plt.show()
print("----------------------------------------------------------------------------------")

df = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/train.csv')
X,Y = preprocess(df)
# print(X)
# print(Y)
n = len(X[0])
onesss = np.ones((len(X),1))
X = np.concatenate((onesss, X), axis = 1)
df_test = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/test.csv')
X_test, Y_test =  preprocess(df_test)
n = len(X_test[0])
onesss = np.ones((len(X_test),1))
X_test = np.concatenate((onesss, X_test), axis = 1)

print("Part one two:")
W = getW_from_formula(X,Y)
print("Coef:",W)
y_t =  X.dot(W)
y_p = X_test.dot(W)
print("test error:",find_error(y_p,Y_test))
print("train error:",find_error(y_t,Y))
E_rms_train = []
E_rms_test = []
X_list = []
for i in range(100):
    W = getW_from_formula(X[0:10*(i+1)], Y[0:10*(i+1)])
    y_t = X[0:10*(i+1)].dot(W)
    y_p = X_test.dot(W)
    # print("test error for",i*10,":",find_error(y_p, Y_test))
    # print("train error for",i*10,":",find_error(y_t, Y[0:10*(i+1)]))
    E_rms_train.append(math.sqrt(2*find_error(y_t, Y[0:10*(i+1)])))
    E_rms_test.append(math.sqrt(2 * find_error(y_p, Y_test)))
    X_list.append(10*(i+1))
plt.plot(X_list, E_rms_test)
plt.plot(X_list, E_rms_train)
plt.show()
print("----------------------------------------------------------------------------------")

df = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/train.csv')
X,Y = preprocess(df)
# print(X)
# print(Y)
df_test = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/test.csv')
X_test, Y_test =  preprocess(df_test)

print("Part two one:")
n = len(X[0])
onesss = np.ones((len(X),1))
X = np.concatenate((onesss, X), axis = 1)
n = len(X_test[0])
onesss = np.ones((len(X_test),1))
X_test = np.concatenate((onesss, X_test), axis = 1)
# w = np.asarray([1] * 10)
w = np.zeros(10)
# print(w)
for i in range(len(w)):
    w[i] = np.random.uniform(0,10)
# print(w)
w, error, error_test = BatchGradientDescent(theta = w,alpha= 0.5 ,X=X, Y = Y,X_test = X_test,Y_test = Y_test, num_iters= 1000, n= n)
print("Coef: ",w)
# print(error)
plt.plot(np.arange(0,1000), error)
plt.plot(np.arange(0,1000), error_test)
plt.show()

print("test error: ",find_error(X_test.dot(w),Y_test))
print("train error: ",find_error(X.dot(w),Y))
print("----------------------------------------------------------------------------------")


print("Part two two:")
df = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/train.csv')
X,Y = preprocess(df)
df_test = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/test.csv')
X_test, Y_test =  preprocess(df_test)
clf = linear_model.SGDRegressor(max_iter=1000, alpha = 0.01)
clf.fit(X, Y)
y_t =  clf.predict(X)
y_p = clf.predict(X_test)
print("Coef: ", clf.coef_, clf.intercept_)
print("test error:",find_error(y_p,Y_test))
print("train error:",find_error(y_t,Y))
print("----------------------------------------------------------------------------------")

print("Part three:")
df = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/train.csv')
X,Y = preprocess(df)
n = len(X[0])
onesss = np.ones((len(X), 1))
X = np.concatenate((onesss, X), axis=1)
df_test = pd.read_csv('/Users/sana/Downloads/Data/Practical 1/test.csv')
X_test, Y_test =  preprocess(df_test)
n = len(X_test[0])
onesss = np.ones((len(X_test), 1))
X_test = np.concatenate((onesss, X_test), axis=1)
landa = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
x = [-4,-3,-2,-1,0,1,2,3,4]
min_sse_err = math.inf
min_sse_err_test = math.inf
best_sse_landa = 0.0
train_sse = []
test_sse = []
saved_w = []
for l in landa:
    w, sse_err = five_fold_cross_validation(X, Y, l)
    train_sse.append(find_SSE_error(X.dot(w),Y, l,w))
    test_sse.append(find_SSE_error(X_test.dot(w),Y_test,l,w))
    if min_sse_err >= sse_err:
        min_sse_err = sse_err
        min_sse_err_test = find_SSE_error(X_test.dot(w),Y_test,l,w)
        best_sse_landa = l
        saved_w = w
plt.plot(x, train_sse)
plt.plot(x, test_sse)
plt.show()
print("best landa:", best_sse_landa)
print("saved w:", saved_w)
print("test error:", find_error(X_test.dot(saved_w),Y_test))
print("train error:", find_error(X.dot(saved_w),Y))

