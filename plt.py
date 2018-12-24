import json
import numpy as np
import matplotlib.pyplot as plt

with open("C:/Users/Administrator/Documents/GitHub/NN-homework/datas", "r") as f:
    test_accuracy, test_cost = json.load(f)
fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(np.arange(0,2 ), test_accuracy, color='#2A6EA6')#
# # ax.set_xlim([training_cost_xmin, num_epochs])
# ax.grid(True)
# ax.set_xlabel('Epoch')
# ax.set_title('Accuracy on the test data')
# plt.show()

ax = fig.add_subplot(111)
ax.plot(np.arange(2), test_cost, color='red')#
# ax.set_xlim([training_cost_xmin, num_epochs])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy on the test data')
plt.show()

