import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class IrisSVM(object):
    def __init__(self):
        self.iris_path = os.path.join(os.getcwd(),'iris.csv')
        self.C = 1000.0

    def load_data(self):
        df = pd.read_csv(self.iris_path)
        data = df.dropna(axis = 0, how ='any')[['sepal_length','sepal_width','species']]
        data = shuffle(data)
        X = data[['sepal_length','sepal_width']].values
        Y = data['species'].values

        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest

    @staticmethod
    def predict_value(X,w,b):
        return (np.dot(X,w) + b).reshape(-1,)

    def train_class(self,label,m):
        Ylabels = np.array([1 if y==label else -1 for y in self.Ytrain])
        Ylabels = Ylabels.reshape(-1,1) * 1.

        X_dash = Ylabels * self.Xtrain
        H = np.dot(X_dash , X_dash.T) * 1.

        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt_matrix(Ylabels.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        w = ((Ylabels * alphas).T @ self.Xtrain).reshape(-1,1)
        S = (alphas > 1e-4).flatten()
        b = Ylabels[S] - np.dot(self.Xtrain[S], w)
        return [w,b[0]]

    def train_model(self):
        m = self.Xtrain.shape[0]
        labels = set(self.Ytrain)
        parameters = []
        for label in list(labels):
            w,b = self.train_class(label,m)
            parameters.append([w,b])
        self.parameters = parameters

    def predict_model(self,X,Y):
        k = len(set(Y))
        m = len(X)
        labels = set(self.Ytrain)
        logits = np.empty([m,k])
        for label in list(labels):
            w,b = self.parameters[label]
            logits[:,label] = IrisSVM.predict_value(X,w,b)
        Ypred = np.argmax(logits, axis=1)
        return np.mean(Ypred == Y)

    def evaluation(self):
        train_acc = self.predict_model(self.Xtrain, self.Ytrain)
        test_acc = self.predict_model(self.Xtest, self.Ytest)
        print("Train accuracy :",train_acc)
        print("Test accuracy :",test_acc)

if __name__ == "__main__":
    classifier = IrisSVM()
    classifier.load_data()
    classifier.train_model()
    classifier.evaluation()