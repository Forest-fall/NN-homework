import numpy as np

class Sigmoid(object):
    def __init__(self, z):
        self.z = z

    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, output):
        return output * (1.0 - output)
    
    def derivative(self):
        return self.z*(1.0 - self.z)


class Tanh(object):
    def forward(self, z):
        return 1.0 - (2.0 / (np.exp(2.0 * z) + 1.0))

    def backward(self, output):
        return 1 - output * output

class Relu(object):
    def forward(self, z):
        return max(0, z)

    def backward(self, output):
        return 1 if output > 0 else 0


class Identity(object):
    def forward(self, z):
        return z

    def backward(self, output):
        return 1