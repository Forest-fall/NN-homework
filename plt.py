import json
import numpy as np
import matplotlib.pyplot as plt

with open("./dnn_loss.3", "r") as f:
    loss = json.load(f)
with open("./dnn_accuracy.3", "r") as f:
    test_accuracy = json.load(f)

fig = plt.figure()

# ax = fig.add_subplot(111)
# ax.plot(np.arange(len(loss)), loss, color='#6593d3')
# ax.grid(True)
# ax.set_xlabel('Epoch')
# ax.set_title('dnn loss with change 3.3.3')
# plt.savefig('./dnn_loss.3.jpg')

ax = fig.add_subplot(111)
ax.plot(np.arange(len(test_accuracy)), test_accuracy, color='red')
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy on testdata')
plt.savefig('./dnn_accuracy.3.jpg')

plt.show()


