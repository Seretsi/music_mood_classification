import numpy as np

class FAI_LinRegClassifier:
    """docstring for FAI_LinRegClassifier"""

    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros((1, 4))
        self.bias = np.zeros((1, 4))

    def forward_pass(self, X, Y):
        loss = (np.dot(self.weights, X) - Y)**2
        return loss
    def backward_pass(self, X, y):
        grad_y = np.multiply(2, np.dot(self.weights, X) - y)
        grad_y = np.multiply(grad_y, X)
        grad_b = np.zeros(2)
        return grad_y, grad_b


    def train(self, X, Y):
        pass
        for i in range(self.iterations):
            for x, y in zip(X, Y):
                loss = self.forward_pass(x, y)
                grad_w, grad_b = self.backward_pass(x, y)
                self.weights = self.weights + grad_w * self.learning_rate
                # self.bias = self.bias + grad_b * self.learning_rate
                if i % 100 == 0:
                    print("Iteration %d, loss %f, grad_w %f" % (i, loss, grad_w))

    def validate(self, X, y):
        pass

    def predict(self, X):
        pass