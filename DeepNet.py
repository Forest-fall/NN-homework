import random
import json

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

class Quadratic(object):
    '''最小二乘'''
    @staticmethod
    def cost_result(a, y, z= None):
        """计算loss"""
        y = 1.0
        return 0.5 * (a - y)**2

    @staticmethod
    def output_delta(a, y, z= None):
        """计算损失函数的导数"""
        return (a - y) * sigmoid_derivative(z)


class LogLikehood(object):
    '''log似然'''
    @staticmethod
    def cost_result(a, y, z= None):
        """分类正确的y=1,所以只需要传入y=1那个类的激活值;别的类y=0无需计算"""
        y = 1.0
        return (- y * np.log(a))

    @staticmethod
    def output_delta(a, y, z= None):
        return (a - y)


class CrossEntropy(object):
    '''交叉熵'''
    @staticmethod
    def cost_result(a, y, z= None):
        return (-y * np.log(a) - (1 - y) * np.log(1-a))

    @staticmethod
    def output_delta(a, y, z= None):
        pass


class FCLayer(object):

    def __init__(self, sizes, cost= LogLikehood):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost
        self.eta = 0.1
        self.delta = [np.zeros((y, 1)) for y in sizes]

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s\n' % (repr(self.weights), repr(self.biases))

    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.biases

    def feedforward(self, x):
        self.input = x
        a = x
        self.activations = [x]
        self.zs = [] 
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = tanh(z)
            # a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = softmax(z)
        # a = sigmoid(z)
        self.zs.append(z)
        self.activations.append(a)
        a = self.activations
        return self.zs[-1], self.activations[-1]

    def backprop(self, y):
        self.lable = y
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta = (self.cost).output_delta(self.activations[-1], y, self.zs[-1]) 

        self.delta[-1] = delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = self.zs[-l]
            activation_derivative = tanh_derivative(z)
            # activation_derivative = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * activation_derivative #Hadamard乘积
            self.delta[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l - 1].transpose()) 
            nabla_b[-l] = delta
        '''计算输入层的误差'''
        self.delta[-self.num_layers] = np.dot(self.weights[-self.num_layers + 1].transpose(), self.delta[-self.num_layers + 1])

        return nabla_w, nabla_b, self.delta

    def update(self, sigma_nabla_w, sigma_nabla_b, mini_batch_size):       
        # self.weights = [(1 - self.eta * (0.1 / 50000)) * w - (self.eta  /mini_batch_size) * nw for w, nw in zip(self.weights, sigma_w)] #正则化
        self.weights = [w - (self.eta  /mini_batch_size) * nw for w, nw in zip(self.weights, sigma_nabla_w)]
        self.biases = [b - (self.eta / mini_batch_size) * nb for b, nb in zip(self.biases, sigma_nabla_b)]
        return self.weights, self.biases
    
    def evaluate(self, data):
        '''返回损失值与正确率'''
        loss = 0
        accuracy_results =[]
        for x,y in data:
            z, a = self.feedforward(x)

            cost = (self.cost).cost_result(a[y], y)
            loss += cost / len(data)
            accuracy_results.append((np.argmax(a), y))
            
        return loss[0], sum(int(x == y) for (x, y) in accuracy_results) / len(data)

    def cnn_cost(self, a, y):
        loss_value = (self.cost).cost_result(a[y], y)
        return loss_value


####functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1.0 - sigmoid(z))

def softmax(z):
    """The softmax funtcion."""
    return np.exp(z) / sum(np.exp(z))

def softmax_derivative(z):
    """Derivative of softmax function."""
    return softmax(z) * (1.0 - softmax(z))


def tanh(z):
    """The tanh function."""
    return 1.0 - (2.0 / (np.exp(2.0 * z) + 1.0))

def tanh_derivative(z):
    """Derivative of the tanh function."""
    return 1.0 - (tanh(z)*tanh(z))

def vectorized_result(j):
    v = np.zeros((10, 1))
    v[j] = 1.0
    return v