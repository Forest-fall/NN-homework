
#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    #print(training_data[0].shape, training_data[1].shape)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_for_dnn():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)       
    training_data = list(training_data)
    #
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    validation_data = list(validation_data)
    #
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    test_data = list(test_data)
    return (training_data, validation_data, test_data)

def load_data_for_cnn():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (28, 28)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    training_data = list(training_data)
    #
    validation_inputs = [np.reshape(x, (28, 28)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    validation_data = list(validation_data)
    #
    test_inputs = [np.reshape(x, (28, 28)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    test_data = list(test_data)

    return (training_data, validation_data, test_data)



def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

load_data_for_cnn()