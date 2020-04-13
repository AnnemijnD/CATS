import numpy as np
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":

    X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
    y = np.array([[1],[1],[1],[0]])
    loss_values = []
    neural = NeuralNetwork(X, y)
    for i in range(1500):
        neural.feedforward()
        neural.backprop()
        loss = compute_loss(neural.output, y)
        loss_values.append(loss)

    print(neural.output)
    print(f" final loss : {loss}")
