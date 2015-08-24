__author__ = 'sridhar'

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    x = np.ones((z.shape[0], z.shape[1]))
    X = x + (np.exp(-1.0 * z))
    Y = np.divide(1, X)
    # print Y
    # print " this is Y" + str(Y.shape)
    return Y


def ComputeLog(val):
    rows = val.shape[0]
    columns = val.shape[1]

    g = np.zeros((rows, columns))
    for i in range(0, rows):
        for j in range(0, columns):
            if (val[i, j] > 0):
                x = np.log(val[i, j]);
            else:
                x = 0;
            g[i, j] = x;

    return g


def preprocess():
    """ Input:
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
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data
    # Your code here
    #
    train_data = np.zeros((49995, 784))
    validation_data = np.zeros((10005, 784))
    train_label = np.zeros((49995, 1))
    validation_label = np.zeros((10005, 1))
    test_data = np.zeros((10000, 784))
    test_label = np.zeros((10000, 1))

    # train_data = np.zeros(())
    # validation_data = np.zeros(())
    # train_label = np.zeros(())
    # validation_label= np.zeros(())
    # test_data = np.zeros(())
    # test_label = np.zeros(())


    k = 0;
    k1 = 0;

    for x in range(0, 10):
        y = 'train' + str(x);
        A = mat.get(y)
        a = range(A.shape[0])
        size_train = (5 * A.shape[0]) / 6
        aperm = np.random.permutation(a)
        train = A[aperm[0:size_train], :]
        validation = A[aperm[size_train:], :]

        for i in range(0, train.shape[0]):
            train_data[k] = train[i, :]
            train_label[k] = x
            k = k + 1

        for i in range(0, validation.shape[0]):
            validation_data[k1] = validation[i, :]
            validation_label[k1] = x
            k1 = k1 + 1

    # normalization

    tvariable = 0
    for z in range(0, 10):
        y = 'test' + str(z);
        A = mat.get(y)
        for i in range(0, A.shape[0]):
            test_data[tvariable] = A[i, :]
            test_label[tvariable] = z
            tvariable = tvariable + 1

    train_data = train_data / 255
    validation_data = validation_data / 255
    test_data = test_data / 255

    # # Feature Classification
    # td =train_data
    # for i in range(0,td.shape[1]):
    #     count = 0
    #     for j in range(0,td.shape[0]):
    #         if td[j,i] == 0:
    #             count = count +1;
    #     for j in range(0,td.shape[0]):
    #         if count>45000:
    #             td[:,i]=[];
    #             test_data[:,i]=[]
    #             validation_data[:,i]=[]

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def oneofK(label, k):
    rows = label.shape[0]
    oneofkmatrix = np.zeros((rows, 10))
    for i in range(0, rows):
        column = label[i, 0]
        oneofkmatrix[i, column] = 1
    return oneofkmatrix


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    td_temp = np.ones((training_data.shape[0], training_data.shape[1] + 1))
    td_temp[:, :-1] = training_data
    training_data = td_temp;

    # print str(training_data.shape) + 'shape of training after append is  '
    # print w1.shape
    hidden_node_op = np.dot(training_data, np.transpose(w1))
    hidden_node_op = sigmoid(hidden_node_op)
    hn_temp = np.ones((hidden_node_op.shape[0], hidden_node_op.shape[1] + 1))
    hn_temp[:, :-1] = hidden_node_op
    Z = hn_temp

    # print str(hidden_node_op.shape) + 'shape of hidden node after append is  '
    # Append a column to hidden_node_op
    output_node_op = np.dot(Z, np.transpose(w2))
    Z1 = sigmoid(output_node_op)

    training_labels = oneofK(training_label, n_class)

    # -----Error
    temp1 = np.multiply(training_labels, ComputeLog(Z1))
    A_var = np.ones((training_labels.shape[0], training_labels.shape[1]))
    A_var = A_var - training_labels
    B_var = np.ones((Z1.shape[0], Z1.shape[1]))
    B_var = B_var - Z1
    temp2 = np.multiply(A_var, ComputeLog(B_var))
    temp3 = temp1 + temp2

    temp3 = np.ndarray.sum(temp3)
    error = np.divide(temp3, (training_data.shape[0] * -1))
    # print "error is" + str(error)

    # -----Regulization
    m1 = np.multiply(w1, w1)
    m2 = np.multiply(w2, w2)
    mat_sum1 = np.ndarray.sum(m1)
    mat_sum2 = np.ndarray.sum(m2)
    mat_sum = mat_sum1 + mat_sum2
    final = np.multiply(mat_sum, lambdaval)
    regulazation = np.divide(final, (2 * training_data.shape[0]))
    # print "regulization is" +str(regulazation)
    obj_val = error + regulazation

    # -------dot product

    obj_grad = np.array([])

    ele_mul = np.multiply(1 - Z, Z)
    ele_res = Z1 - training_labels
    ele2_mul = np.dot(ele_res, w2)
    grad_mul = np.multiply(ele_mul, ele2_mul)
    grad_tran = np.transpose(grad_mul)

    grad_w1_res = np.dot(grad_tran, training_data)
    grad_w1 = np.zeros((grad_w1_res.shape[0], grad_w1_res.shape[1] - 1))
    grad_w1 = grad_w1_res[:-1, :]

    mat1 = Z1 - training_labels
    mat2 = np.transpose(mat1)
    mat3 = np.dot(mat2, Z)
    grad_w2 = mat3

    mat4 = grad_w1 + np.multiply(lambdaval, w1)
    mat5 = grad_w2 + np.multiply(lambdaval, w2)
    grad_w1 = mat4 / training_data.shape[0]
    grad_w2 = mat5 / training_data.shape[0]

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    return obj_val, obj_grad


def oneofKdecode(oneofK):
    size = oneofK.shape[0]
    # print size
    labels = np.zeros((size, 1))
    # print labels.shape
    for i in range(0, size):
        labels[i] = np.argmax(oneofK[i, :], axis=0)

    return labels


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1,
w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here


    data_temp = np.ones((data.shape[0], data.shape[1] + 1))
    data_temp[:, :-1] = data
    data = data_temp;

    X = np.dot(data, np.transpose(w1))
    sigm_out = sigmoid(X)

    temp = np.ones((sigm_out.shape[0], sigm_out.shape[1] + 1))
    temp[:, :-1] = sigm_out
    sigm_out = temp;

    A = sigmoid(np.dot(sigm_out, np.transpose(w2)));
    labels = oneofKdecode(A);
    # print labels
    # print " labels ki "
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
