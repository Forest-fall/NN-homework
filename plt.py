import json
import numpy as np
import matplotlib.pyplot as plt

with open("C:/Users/Administrator/Documents/GitHub/NN-homework/3_Q_sigmoid_init", "r") as f:
    test_accuracy = json.load(f)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(30), test_accuracy, color='#FF4500')
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy on the test data with 3 layers + Quadratic + sigmoid + initialize weight')
plt.savefig('./3_Q_sigmoid_init.jpg')
plt.show()


