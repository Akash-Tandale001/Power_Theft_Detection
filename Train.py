import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Conv2D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import DataProcessing





tf.random.set_seed(1234)
epochs_number = 1  # number of epochs for the neural networks
test_set_size = 0.1  # percentage of the test size comparing to the whole dataset
oversampling_flag = 0  # set to 1 to over-sample the minority class
oversampling_percentage = 0.2  # percentage of the minority class after the oversampling comparing to majority class

# Definition of functions
def read_data():
    rawData = pd.read_csv('preprocessedR.csv')

    # Setting the target and dropping the unnecessary columns
    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    print('Normal Consumers:                    ', y[y['FLAG'] == 0].count()[0])
    print('Consumers with Fraud:                ', y[y['FLAG'] == 1].count()[0])
    print('Total Consumers:                     ', y.shape[0])
    print("Classification assuming no fraud:     %.2f" % (y[y['FLAG'] == 0].count()[0] / y.shape[0] * 100), "%")

    # columns reindexing according to dates
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=test_set_size, random_state=0)
    print("Test set assuming no fraud:           %.2f" % (y_test[y_test == 0].count() / y_test.shape[0] * 100), "%\n")

    # Oversampling of minority class to encounter the imbalanced learning
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)
        print("Oversampling statistics in training set: ")
        print('Normal Consumers:                    ', y_train[y_train == 0].count())
        print('Consumers with Fraud:                ', y_train[y_train == 1].count())
        print("Total Consumers                      ", X_train.shape[0])

    return X_train, X_test, y_train, y_test


def results(y_test, prediction):
    print("Accuracy", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("AUC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def ANN(X_train, X_test, y_train, y_test):
    print('Artificial Neural Network:')
    # for i in range(4,100,3):
    #     print("Epoch:",i)

    # Model creation
    model = Sequential()
    model.add(Dense(1000, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(X_train, y_train, validation_split=0, epochs=i, shuffle=True, verbose=0)
    model.fit(X_train, y_train, validation_split=0, epochs=epochs_number, shuffle=True, verbose=1)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    model.summary()
    results(y_test, prediction)


def CNN1D(X_train, X_test, y_train, y_test):
    print('1D - Convolutional Neural Network:')

    # Transforming the dataset into tensors
    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    # Model creation
    model = Sequential()
    model.add(Conv1D(100, kernel_size=7, input_shape=(1034, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs_number, validation_split=0, shuffle=False, verbose=1)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    model.summary()
    results(y_test, prediction)




def LR(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')
    '''
    # Parameters selection 
    param_grid = {'C': [0.1,1,10,100],'solver': ['newton-cg', 'lbfgs']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000,random_state=0), param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)
    df = pd.DataFrame(grid.cv_results_)
    print(df[['param_C', 'param_solver', 'mean_test_score', 'rank_test_score']])
    '''
    model = LogisticRegression(C=1000, max_iter=1000, n_jobs=-1, solver='newton-cg')
    model.fit(X_train, y_train)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    results(y_test, prediction)


def DT(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    results(y_test, prediction)


def RF(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    '''
    # Parameters selection 
    param_grid = {'n_estimators':[10,100,1000]}
    grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)
    df = pd.DataFrame(grid.cv_results_)
    print(df[['param_criterion', 'mean_test_score', 'rank_test_score']])
    '''

    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='auto',  # max_depth=10,
                                   random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    results(y_test, prediction)


def SVM(X_train, X_test, y_train, y_test):
    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    results(y_test, prediction)


# ----Main----
def processTraining():
    X_train, X_test, y_train, y_test = read_data()

    # Uncomment any model to test
    #ANN(X_train, X_test, y_train, y_test)
    CNN1D(X_train, X_test, y_train, y_test)
    #RF(X_train, X_test, y_train, y_test)
    #LR(X_train, X_test, y_train, y_test)
    #DT(X_train, X_test, y_train, y_test)
    #SVM(X_train, X_test, y_train, y_test)


