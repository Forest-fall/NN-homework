import mnist_loader
import ConvNN
import DeepNet

import numpy as np
import random
import matplotlib.pyplot as plt
import json


training_data, validation_data, test_data = mnist_loader.load_data_for_cnn()
training_data = training_data
test_data = test_data
learning_rate = 0.1

c1 = ConvNN.ConvLayer(28, 28, 1, 5, 5, 6, 2, 1)
p2 = ConvNN.MaxPoolingLayer(28, 28, 6, 2, 2, 2)
c3 = ConvNN.ConvLayer(14, 14, 6, 5, 5, 16, 0, 1)
p4 = ConvNN.MaxPoolingLayer(10, 10, 16, 2, 2, 2)
c5 = ConvNN.ConvLayer(5, 5, 16, 5, 5, 120, 0, 1)
fc = DeepNet.FCLayer([120, 84, 10])

num = 0

def fc_in(array_in):
   '''c5卷积层的输出是(120,1,1),而全连接层的输入是(120,1),改一下shape'''
   L = []
   for i in np.nditer(array_in):
      L.append(i)
   array_out = array_in.reshape(120, 1)
   return array_out

n_test = len(test_data)
n_train = len(training_data)
mini_batch_size = 10
eta = 0.1

#用于存放数据绘图
Loss = []
Accuracy = []

#一次迭代 epoch = 1
for epoch in range(30):
   '''每一次迭代都打乱一次数据'''
   random.shuffle(training_data)
   mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]

   for mini_batch in mini_batches:

      fc_sigma_nabla_w = [np.zeros(w.shape) for w in fc.get_weights()]
      fc_sigma_nabla_b = [np.zeros(b.shape) for b in fc.get_biases()]

      c5_sigma_nabla_w = [np.zeros(filter.get_weights().shape) for filter in c5.filters]
      c5_sigma_nabla_b = [0] * c5.filter_number

      c3_sigma_nabla_w = [np.zeros(filter.get_weights().shape) for filter in c3.filters]
      c3_sigma_nabla_b = [0] * c3.filter_number

      c1_sigma_nabla_w = [np.zeros(filter.get_weights().shape) for filter in c1.filters]
      c1_sigma_nabla_b = [0] * c1.filter_number

      num += 1
      for x,y in mini_batch:

         # output_z, output_a = net.feedforward(x)
         # nabla_w, nabla_b = net.backprop(y)
         # sigma_w = [sw + nw for sw, nw in zip(sigma_w, nabla_w)]
         # sigma_b = [sb + nb for sb, nb in zip(sigma_b, nabla_b)]
         # update_w, update_b = net.update(sigma_w, sigma_b, mini_batch_size)

         #forward
         c1.forward(x)
         c1_output = c1.output_array
         # print(c1_output.shape)

         p2.forward(c1_output)
         p2_output = p2.output_array
         # print(p2_output.shape)
   
         c3.forward(p2_output)
         c3_output = c3.output_array
         # print(c3_output.shape)
         
         p4.forward(c3_output)
         p4_output = p4.output_array
         # print(p4_output.shape)

         c5.forward(p4_output)
         c5_output = c5.output_array
         # print(c5_output.shape)
         c5_output = fc_in(c5_output)
         # print(c5_output.shape)

         fc_output_z, fc_output_a = fc.feedforward(c5_output)

         #backward
         fc_nabla_w, fc_nabla_b, fc_delta = fc.backprop(y)
      
         c5_SensitivityMap = fc_delta[0].reshape(120, 1, 1)
         c5_delta_map = c5.backward(c5_SensitivityMap)
      
         p4_SensitivityMap = c5_delta_map
         p4_delta_map = p4.backward(p4_SensitivityMap) 

         c3_SensitivityMap = p4_delta_map
         c3_delta_map = c3.backward(c3_SensitivityMap)

         p2_SensitivityMap = c3_delta_map
         p2_delta_map = p2.backward(p2_SensitivityMap) 

         c1_SensitivityMap = p2_delta_map
         c1_delta_map = c1.backward(c1_SensitivityMap)

         '''sigma parameters'''
         fc_sigma_nabla_w = [sw + nw for sw, nw in zip(fc_sigma_nabla_w, fc_nabla_w)]
         fc_sigma_nabla_b = [sb + nb for sb, nb in zip(fc_sigma_nabla_b, fc_nabla_b)]

         c5_sigma_nabla_w = [sw + filter.get_nabla_weights() for sw, filter in zip(c5_sigma_nabla_w, c5.filters)]
         c5_sigma_nabla_b = [sb + filter.get_nabla_bias() for sb, filter in zip(c5_sigma_nabla_b, c5.filters)]

         c3_sigma_nabla_w = [sw + filter.get_nabla_weights() for sw, filter in zip(c3_sigma_nabla_w, c3.filters)]
         c3_sigma_nabla_b = [sb + filter.get_nabla_bias() for sb, filter in zip(c3_sigma_nabla_b, c3.filters)]

         c1_sigma_nabla_w = [sw + filter.get_nabla_weights() for sw, filter in zip(c1_sigma_nabla_w, c1.filters)]
         c1_sigma_nabla_b = [sb + filter.get_nabla_bias() for sb, filter in zip(c1_sigma_nabla_b, c1.filters)]

      #update parameters
      fc_w, fc_b = fc.update(fc_sigma_nabla_w, fc_sigma_nabla_b, mini_batch_size)
      c5_filters = c5.update(c5_sigma_nabla_w, c5_sigma_nabla_b, mini_batch_size)
      c3_filters = c3.update(c3_sigma_nabla_w, c3_sigma_nabla_b, mini_batch_size)
      c1_filters = c1.update(c1_sigma_nabla_w, c1_sigma_nabla_b, mini_batch_size)

   losses = 0
   accuracy_results =[]
   for x, y  in test_data:
      #forward
      c1.forward(x)
      c1_output = c1.output_array
      p2.forward(c1_output)
      p2_output = p2.output_array
      c3.forward(p2_output)
      c3_output = c3.output_array
      p4.forward(c3_output)
      p4_output = p4.output_array
      c5.forward(p4_output)
      c5_output = c5.output_array
      c5_output = fc_in(c5_output)
      fc_output_z, fc_output_a = fc.feedforward(c5_output)

      loss = fc.cnn_cost(fc_output_a, y) 
      losses += loss / len(test_data)
      accuracy_results.append((np.argmax(fc_output_a), y))
   test_accuracy = sum(int(x == y) for (x, y) in accuracy_results) / len(test_data)
   print("Epoch:{0}, loss:{1}, accuracy:{2:.2%}".format(epoch, losses, test_accuracy))
   # Loss.append(losses)
   # Accuracy.append(test_accuracy)
   # with open("./cnn_loss", "w") as f:
   #    json.dump(Loss, f)

#每一次迭代完成，计算一次正确率
      # fig = plt.figure()
      # plt.imshow(c1_delta_map[0])
      # plt.axis('off')
      # plt.show()

      # d = c1_delta_map

      


      
      

# print("Epoch 0 : {0}".format(net.evaluate_accuracy(test_data)))

       
       

   

