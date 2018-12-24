import json
import numpy as np
import matplotlib.pyplot as plt

with open("C:/Users/Forest-fall/Documents/GitHub/NN-homework/test_accuracy", "r") as f:
    test_accuracy = json.load(f)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(30), test_accuracy, color='#6593d3')
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy on the test data with one hidden layer')
plt.savefig('./test_accuracy_With1Hidden.jpg')
plt.show()


