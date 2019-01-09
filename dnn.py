#最初的深度神经网络的版本
import random
import json

# Third-party libraries
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
class Quadratic(object):

    @staticmethod
    def cost_result(a, y, z= None):
        """return the output_result of this cost function"""
        return 0.5 * (a - y)**2

    @staticmethod
    def output_delta(a, y, z= None):
        """return the derivative of this cost function"""
        return (a - y) * sigmoid_derivative(z)


class LogLikehood(object):

    @staticmethod
    def cost_result(a, y, z= None):
        """return the output_result of this cost function"""
        return (- y * np.log(a))

    @staticmethod
    def output_delta(a, y, z= None):
        """return the derivative of this cost function"""
        return (a - y)


class CrossEntropy(object):

    @staticmethod
    def cost_result(a, y, z= None):
        """return the output_result of this cost function"""
        return (-y * np.log(a) - (1 - y) * np.log(1-a))

    @staticmethod
    def output_delta(a, y, z= None):
        """return the derivative of this cost function"""
        pass


class FCLayer(object):

    def __init__(self, sizes, cost= LogLikehood):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            #a = sigmoid(np.dot(w, a)+b)
            z = np.dot(w, a) + b
            # print("mean:", np.mean(z))
            # print("std:", np.std(z))
            a = tanh(np.dot(w, a) + b)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: 
            n_test = len(test_data)
        n_train = len(training_data)
        test_error = []
        log_cost = []
        for j in range(epochs):
            #打乱顺序
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                # self.update_mini_batch(mini_batch, eta)
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            # test_cost.append(self.cost(test_data).tolist())
            # test_error.append(self.evaluate(test_data) / n_test)
        # with open("C:/Users/Forest-fall/Documents/GitHub/NN-homework/test_accuracy","w") as f:
        #     json.dump(test_error, f)

        #version2
        # for j in range(epochs):
        #     random.shuffle(training_data)

        #     batch_epoch = 0
        #     for k in range(0, n_train, mini_batch_size):
        #         mini_batch = training_data[k: k + mini_batch_size]
        #         batch_epoch += 1
        #         self.update_mini_batch(mini_batch, eta, test_data, j, batch_epoch, n_test, test_error, log_cost)
     

    def update_mini_batch(self, mini_batch, eta,         ):#test_data, j, batch_epoch, n_test, test_error, log_cost

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #log_cost
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta  /len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        #如果每一批都看一次正确率呢？？
        # if test_data:
        #     print("Epoch {0}, batch {1}: {2} / {3}".format(j, batch_epoch, self.evaluate(test_data), n_test))
        # else:
        #     print("Epoch {0} complete".format(j))
        # test_error.append(self.evaluate(test_data) / n_test)
        # with open("C:/Users/Forest-fall/Documents/GitHub/NN-homework/batch_accuracy", "w") as f:
        #     json.dump(test_error, f)

    def backprop(self, x, y): #,      log_cost
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            #activation = sigmoid(z)
            activation = tanh(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        #计算Loss
        # y_index = np.argmax(y)
        # a = activations[-1][y_index]
        # loss = (self.cost).cost_result(a,1).tolist()[0]
        # log_cost.append(loss)
        # with open("C:/Users/Forest-fall/Documents/GitHub/NN-homework/log_cost", "w") as f:
        #     json.dump(log_cost, f)

        # backward pass
        #delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        
        delta = (self.cost).output_delta(activations[-1], y, zs[-1]) #activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
        for l in range(2, self.num_layers):
            z = zs[-l]
            #activation_derivative = sigmoid_derivative(z)
            activation_derivative = tanh_derivative(z)
            #Hadamard乘积
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * activation_derivative 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        "caluculate accuracy on test data"
        test_results =[]
        for x, y in test_data:
            y_predict = np.argmax(self.feedforward(x))
            test_results.append((y_predict, y))
        # test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    
    # def cost(self, data):
    #     cost = 0
    #     for x, y in data:
    #         cost += -y * np.log(self.feedforward(x)) / len(data)
    #     return cost
            


    # def cost_derivative(self, output_activations, y):
    #     """Return the vector of partial derivatives  (partial C_x \partial a )for the output activations."""
    #     return (output_activations - y)
    

#### Miscellaneous functions
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