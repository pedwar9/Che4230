#Import packages
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier

class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self):
        #Load data
        data = pd.read_excel("Data/Know_Normal_Operation.xlsx", header = 0).dropna()

        data = data.to_numpy()

        return data

    def split_data(self, data):

        x = data[:, :-1]
        y = data[:, -1]
        y = y.astype(int)

        #Data preprocessing
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        #Create train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.25,
                                                            shuffle=True)

        return X_train, X_test, y_train, y_test

    def split_data_fault(self, data):

        x = data[:, :-1]
        y = data[:, -1]
        y = y.astype(int)

        #Data preprocessing
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        #Create train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.25,
                                                            shuffle=True)

        return X_train, X_test, y_train, y_test
        
    def load_data_fault(self):
        #Load data
        data = pd.read_excel("Data/RawData.xlsx", header = 0).dropna()

        data = data.to_numpy()

        return data


preprocessor = DataPreprocessing()
knowndata = preprocessor.load_data()
#print(data)
X_train, X_test, y_train, y_test = preprocessor.split_data(knowndata)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

faultdata = preprocessor.load_data_fault()
X_trainf, X_testf, y_trainf, y_testf = preprocessor.split_data_fault(faultdata)

y_pred = clf.predict(X_testf)

score = accuracy_score(y_testf,y_pred)

print("SVM Accuracy =",score)

classifier=MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter=300, activation='relu',solver='adam',random_state=1)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_trainf)

score2 = accuracy_score(y_trainf,y_pred)
print("ANN Accuracy =",score2)



