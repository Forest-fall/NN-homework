#最初的cnn测试版本
import numpy as np
import matplotlib.pyplot as plt

import ConvNN
import DNN
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_for_cnn()

#forward
c1 = ConvNN.ConvLayer(28, 28, 1, 5, 5, 6, 2, 1, 0.001)
input_array = training_data[0][0]
c1.forward(input_array)
c1_output = c1.output_array
print(c1_output.shape)

p2 = ConvNN.MaxPoolingLayer(28, 28, 6, 2, 2, 2)
p2.forward(c1_output)
p2_output = p2.output_array
print(p2_output.shape)

c3 = ConvNN.ConvLayer(14, 14, 6, 5, 5, 16, 0, 1, 0.001)
input_array = p2_output
c3.forward(input_array)
c3_output = c3.output_array
print(c3_output.shape)

p4 = ConvNN.MaxPoolingLayer(10, 10, 16, 2, 2, 2)
p4.forward(c3_output)
p4_output = p4.output_array
print(p4_output.shape)

c5 = ConvNN.ConvLayer(5, 5, 16, 5, 5, 120, 0, 1, 0.001)
input_array = p4_output
c5.forward(input_array)
c5_output = c5.output_array
print(c5_output.shape)
L = []
for i in np.nditer(c5_output):
    L.append(i)
c5_output = np.array(L).reshape(120,1)
print(c5_output.shape)




# input_array = training_data[0][0]
# cnn.forward(input_array)
# output = cnn.output_array[0]
# print(np.argmax(training_data[0][1]))
# fig = plt.figure()
# plt.subplot(241)
# plt.imshow(cnn.output_array[0])
# plt.axis('off')

# plt.subplot(242)
# plt.imshow(cnn.output_array[1])
# plt.axis('off')

# plt.subplot(243)
# plt.imshow(cnn.output_array[2])
# plt.axis('off')

# plt.subplot(244)
# plt.imshow(cnn.output_array[3])
# plt.axis('off')

# plt.subplot(245)
# plt.imshow(cnn.output_array[4])
# plt.axis('off')

# plt.subplot(246)
# plt.imshow(cnn.output_array[5])
# plt.axis('off')

# plt.subplot(247)
# plt.imshow(input_array)
# plt.axis('off')
# plt.show()