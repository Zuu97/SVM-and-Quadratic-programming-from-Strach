import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class IrisSVM(object):
    def __init__(self):
        self.iris_path = os.path.join(os.getcwd(),'iris.csv')
        self.grid_parameters = {'kernel': ['rbf'], 'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001]}

    def load_data(self):
        df = pd.read_csv(self.iris_path)
        data = df.dropna(axis = 0, how ='any')[['sepal_length','sepal_width','species']]
        data = shuffle(data)
        X = data[['sepal_length','sepal_width']]
        Y = data['species']
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest

    @staticmethod
    def modified_cv(input_train_len, input_test_len):
        yield (np.array(range(input_train_len)),
            np.array(range(input_train_len, input_train_len + input_test_len)))

    def train_model(self):
        classifier = SVC()
        input_train_len = len(self.Xtrain)
        input_test_len  = len(self.Xtest)
        X = np.concatenate((self.Xtrain, self.Xtest), axis=0)
        Y = np.concatenate((self.Ytrain, self.Ytest), axis=0)
        self.grid = GridSearchCV(classifier,
                                 self.grid_parameters,
                                 refit = True,
                                 verbose=3,
                                 cv=IrisSVM.modified_cv(input_train_len, input_test_len))
        self.grid_history = self.grid.fit(X, Y)

    def evaluation(self):
        test_score = self.grid_history.cv_results_['split0_test_score']
        print("Test Scores: ",test_score)

if __name__ == "__main__":
    classifier = IrisSVM()
    classifier.load_data()
    classifier.train_model()
    classifier.evaluation()