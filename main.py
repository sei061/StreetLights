import numpy as np
import time
np.random.seed(1)

def sig(x):
    return 1/(1 + np.exp(-x))

def sig_grad(x):
    return x * (1 - x)

streetlights = np.array([[1,0,1],
                         [0,1,1],
                         [0,0,1],
                         [1,1,1],
                         [0,1,1],
                         [1,0,1]])

walk_vs_stop = np.array([[0], [1], [0], [1], [1], [0]])
X,y = streetlights, walk_vs_stop
hidden_nodes = 8

epochs = 300  # number of iterations to go through the network

lr = 0.001    # how much we change the weights of the network each iteration

ws_1 = np.random.rand(X.shape[1], hidden_nodes) - 0.5
ws_2 = np.random.rand(ws_1.shape[1], hidden_nodes) - 0.5
ws_3 = np.random.rand(hidden_nodes, y.shape[1]) - 0.5

for epoch in range(epochs):  # number of training iterations, or times to change the weights of the nn
    for i in range(X.shape[0]):  # for all samples in X, each streetlight
        layer_in = X[i:i + 1]

        # forward pass/prediction
        layer_1 = sig(layer_in.dot(ws_1))
        layer_2 = sig(layer_1.dot(ws_2))

        layer_out = layer_2.dot(ws_3)

        # calc error/distance (how far are we from goal)
        delta_3 = layer_out - y[i:i + 1]

        # calc the the error each node in prev layer contributed

        delta_2 = delta_3.dot(ws_3.T) * sig_grad(layer_2)
        delta_1 = delta_2.dot(ws_2.T) * sig_grad(layer_1)

        # update weights
        ws_3 -= lr * (layer_2.T.reshape(hidden_nodes, 1).dot(delta_3))
        ws_2 -= lr * (layer_1.T.reshape(hidden_nodes, 1).dot(delta_2))
        ws_1 -= lr * (layer_in.T.reshape(X.shape[1], 1).dot(delta_1))

    if epoch % 10 == 0:
        error = delta_2 ** 2
        print(round(error[0][0], 6))  # , end='\r')