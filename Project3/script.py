import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.svm import SVC


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection
    """

    mat = loadmat('mnist_all.mat');  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10 * n_validation;

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature));
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :];

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1));
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1));

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature));
    train_label = np.zeros((n_train, 1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0];
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :];
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1));
        temp = temp + size_i - n_validation;

    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0];
    test_data = np.zeros((n_test, n_feature));
    test_label = np.zeros((n_test, 1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0];
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i));
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1));
        temp = temp + size_i;

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0);
    index = np.array([]);
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:, index.astype(int)];
    validation_data = validation_data[:, index.astype(int)];
    test_data = test_data[:, index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data / 255.0;
    validation_data = validation_data / 255.0;
    test_data = test_data / 255.0;

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));


def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """

    train_data = args[0]
    labeli = args[1]

    n_data = train_data.shape[0];
    n_feature = train_data.shape[1];
    error = 0;
    error_grad = np.zeros((n_feature + 1, 1));

    ##################
    # YOUR CODE HERE #
    ##################

    train_data = np.concatenate((train_data, (np.ones((n_data, 1)))), axis=1)
    Y = sigmoid(np.dot(train_data, initialWeights))
    error = np.multiply(np.sum(np.multiply(labeli, np.log(Y)) + np.multiply((1 - labeli), np.log(1 - Y))), (-1))
    error_grad = np.dot(np.transpose(train_data), (Y - labeli))

    error_grad = error_grad.flatten()
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    label = np.zeros((data.shape[0], 1))
    data = np.concatenate((data, (np.ones((data.shape[0], 1)))), axis=1)
    Y = sigmoid(np.dot(data, W));
    label = np.amax(Y, axis=1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();

# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));
for i in range(n_class):
    T[:, i] = (train_label == i).astype(int).ravel();

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class));
initialWeights = np.zeros((n_feature + 1, 1));
opts = {'maxiter': 50};
for i in range(n_class):
    labeli = T[:, i].reshape(n_train, 1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,));

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

print('\n\n--------------Linear _ Default Parameters-------------------\n\n')
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n\n--------------RBF kernel gamma =1 ----------------------\n\n')
clf = SVC(kernel='rbf', gamma=1)
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n\n--------------Modelling SVM with RBF kernel(default parameters)----------------------\n\n')
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n\n--------------RBF kernel for C =1 --------------------\n\n')
clf = SVC(C=1.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 10 --------------------\n\n')
clf = SVC(C=10.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 20 --------------------\n\n')
clf = SVC(C=20.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 30 --------------------\n\n')
clf = SVC(C=30.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 40 --------------------\n\n')
clf = SVC(C=40.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 50 --------------------\n\n')
clf = SVC(C=50.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 60 --------------------\n\n')
clf = SVC(C=60.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 70 --------------------\n\n')
clf = SVC(C=70.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 80 --------------------\n\n')
clf = SVC(C=80.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 90 --------------------\n\n')
clf = SVC(C=90.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n--------------RBF kernel for C = 100 --------------------\n\n')
clf = SVC(C=100.0, kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%\n')

print('\n\n-----------------END-------------------\n\n')
