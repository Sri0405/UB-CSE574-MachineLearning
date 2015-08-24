import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import inv


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    # IMPLEMENT THIS METHOD

    N = X.shape[0]
    d = X.shape[1]
    labels = y.reshape(y.size)
    clabels = np.unique(labels)
    clabelsiz = clabels.shape[0]
    means = np.zeros((d, clabelsiz))

    for i in range(clabelsiz):
        means[:, i] = np.mean(X[labels == clabels[i]], axis=0)

    covmat = np.cov(X, rowvar=0)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    N = X.shape[0]
    d = X.shape[1]

    labels = y.reshape(y.size)
    clabels = np.unique(labels)
    clabelsiz = clabels.shape[0]
    means = np.zeros((d, clabelsiz))
    covmats = [np.zeros((d, d))] * clabelsiz

    for i in range(clabelsiz):
        means[:, i] = np.mean(X[labels == clabels[i]], axis=0)
        covmats[i] = np.cov(X[labels == clabels[i]], rowvar=0)

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD
    detcovmat = np.linalg.det(covmat)
    invcovmat = np.linalg.inv(covmat)

    out = np.zeros((Xtest.shape[0], means.shape[1]))
    siz = means.shape[1]

    for i in range(siz):
        denominatr = (np.sqrt(np.pi * 2) * (np.power(detcovmat, 2)))
        temp_1 = Xtest - means[:, i]
        temp_1_t = np.transpose(temp_1)
        temp_2 = np.dot(invcovmat, temp_1_t)
        temp_2_t = np.transpose(temp_2)

        temp_4 = np.sum(temp_1 * temp_2_t, 1)
        numerator = np.exp(-0.5 * temp_4)
        out[:, i] = numerator / denominatr

    Label = np.argmax(out, 1)
    Label = Label + 1
    ytest = ytest.reshape(ytest.size)

    acc = 100 * np.mean(Label == ytest)
    return acc


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD


    out = np.zeros((Xtest.shape[0], means.shape[1]))
    siz = means.shape[1]
    # pdf[:,i] = np.exp(-0.5*temp_4)/denominator

    for i in range(siz):
        invcovmat = np.linalg.inv(covmats[i])
        detcovmat = np.linalg.det(covmats[i])

        denominatr = (np.sqrt(np.pi * 2) * (np.power(detcovmat, 2)))
        temp_1 = Xtest - means[:, i]
        temp_1_t = np.transpose(temp_1)
        temp_2 = np.dot(invcovmat, temp_1_t)
        temp_2_t = np.transpose(temp_2)

        temp_4 = np.sum(temp_1 * temp_2_t, 1)
        numerator = np.exp(-0.5 * temp_4)
        out[:, i] = numerator / denominatr

    Label = np.argmax(out, 1)
    Label = Label + 1
    ytest = ytest.reshape(ytest.size)

    acc = 100 * np.mean(Label == ytest)

    return acc


def learnOLERegression(X, y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD

    N = X.shape[0]
    d = X.shape[1]

    w = np.zeros((d, 1))

    Xtrans = np.transpose(X)

    # w_MLE = inverse(Xtrans * X) * transpose(X) * y;

    temp_1 = np.dot(Xtrans, X)
    Inv = inv(temp_1)
    temp_2 = np.dot(Xtrans, y)

    w = np.dot(Inv, temp_2)
    # print " OLE REgression weights"
    # print w
    print "weights mean ole regression"
    print np.mean(w)
    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    # IMPLEMENT THIS METHOD                                                   

    N = X.shape[0]
    d = X.shape[1]

    Trans_X = np.transpose(X)
    Identity_mat = np.identity(d)
    temp_4 = inv(np.dot(Trans_X, X) + lambd * N * Identity_mat)
    temp_5 = np.dot(Trans_X, y)
    w = np.dot(temp_4, temp_5)

    # print "ridge regression weights"
    # print w
    print "weight mean ridge regression"
    print np.mean(w)
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    ytest_transpose = np.transpose(ytest)
    xtest_trannpose = np.transpose(Xtest)
    w_transpose = np.transpose(w)
    temp1 = np.dot(w_transpose, xtest_trannpose)

    temp_5 = np.subtract(ytest_transpose, temp1)
    temp_6 = np.multiply(temp_5, temp_5)
    temp_7 = np.sum(temp_6)
    temp_8 = np.sqrt(temp_7)
    rmse = temp_8 / Xtest.shape[0]
    return rmse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    error = 0
    w_transpose = np.transpose(w)
    X_transpose = np.transpose(X)
    y_transpose = np.transpose(y)

    temp_1 = np.subtract(y_transpose, np.dot(w_transpose, X_transpose))
    temp_2 = np.multiply(temp_1, temp_1)
    sum = np.sum(temp_2)

    N = X.shape[0]
    val1 = sum
    temp_3 = np.dot(w_transpose, w)
    val2 = np.multiply(lambd, temp_3)
    # print val1
    # print val2
    error = val1 / (np.multiply(2, N)) + val2 / 2

    error_grads = np.add(np.subtract(np.multiply(np.multiply(lambd, N), w), np.dot(y_transpose, X)),
                         np.dot(w_transpose, np.dot(X_transpose, X)))
    error_grad = np.squeeze(np.asarray(error_grads)) / N
    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD

    n_rows = x.shape[0]
    Xd = np.ones((n_rows, p + 1));
    for i in range(1, p + 1):
        Xd[:, i] = x ** i;

    return Xd


# Main script

# Problem 1
# load the sample data
X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# Problem 2

X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')
# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('RMSE without intercept ' + str(mle))
print('RMSE with intercept ' + str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    rmses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
plt.plot(lambdas, rmses3)

# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k, 1))
opts = {'maxiter': 100}  # Preferred value.
w_init = np.zeros((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1], 1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1, Xtest_i, ytest)
    i = i + 1
plt.plot(lambdas, rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    rmses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lamda_opt)
    rmses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)
plt.plot(range(pmax), rmses5)
plt.legend(('No Regularization', 'Regularization'))
