import json
import numpy as np
import matplotlib.pyplot as plt

with open("./cnn_loss", "r") as f:
    loss = json.load(f)
# with open("./dnn_accuracy", "r") as f:
#     test_accuracy = json.load(f)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(len(loss)), loss, color='#6593d3')
ax1.grid(True)
ax1.set_xlabel('Epoch')
ax1.set_title('cnn loss')
plt.savefig('./cnn_loss.jpg')

# ax2 = fig.add_subplot(212)
# ax2.plot(np.arange(3), test_accuracy, color='#6593d3')
# ax2.grid(True)
# ax2.set_xlabel('Epoch')
# ax2.set_title('Accuracy on testdata')
# plt.savefig('./dnn_accuracy.jpg')

plt.show()


