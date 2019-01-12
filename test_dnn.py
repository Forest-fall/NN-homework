import DeepNet
import mnist_loader
import numpy as np
import json
import random

training_data, validation_data, test_data = mnist_loader.load_data_for_dnn()
net = DeepNet.FCLayer([784, 100, 100, 100, 10])

# test_data = test_data[0:10]
n_test = len(test_data)
n_train = len(training_data)
mini_batch_size = 10

#用于存放数据绘图
Loss = []
Accuracy = []

for epoch in range(30):
    '''每一次迭代都打乱一次数据'''
    random.shuffle(training_data)
    mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]

    for mini_batch in mini_batches:
        sigma_w = [np.zeros(w.shape) for w in net.get_weights()]
        sigma_b = [np.zeros(b.shape) for b in net.get_biases()]
        for x, y in mini_batch:
            output_z, output_a = net.feedforward(x)
            nabla_w, nabla_b, detla = net.backprop(y)
            sigma_w = [sw + nw for sw, nw in zip(sigma_w, nabla_w)]
            sigma_b = [sb + nb for sb, nb in zip(sigma_b, nabla_b)]
        update_w, update_b = net.update(sigma_w, sigma_b, mini_batch_size)
        
    loss, test_accuracy = net.evaluate(test_data)
    print("Epoch:{0}, loss:{1}, accuracy:{2:.2%}".format(epoch, loss, test_accuracy))
    Loss.append(loss)
    Accuracy.append(test_accuracy)
    with open("./dnn_loss", "w") as f:
        json.dump(Loss, f)
    with open("./dnn_accuracy", "w") as f:
        json.dump(Accuracy, f)
    