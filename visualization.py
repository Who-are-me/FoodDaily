import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


csv_data = pd.read_csv('data.csv')
fig, axs = plt.subplots(2,3)

fig.suptitle('Metrics')

axs[0][0].scatter(csv_data['item'], csv_data['accuracy'])
axs[0][0].set_title("Accuracy")

axs[0][1].scatter(csv_data['item'], csv_data['precision'])
axs[0][1].set_title("Precision")

axs[0][2].plot(csv_data['item'], csv_data['recall'])
axs[0][2].set_title("Recall")

axs[1][0].scatter(csv_data['item'], csv_data['f0'])
axs[1][0].set_title("F(0)")

axs[1][1].plot(csv_data['item'], csv_data['accuracy_sum'])
axs[1][1].set_title("Accuracy while run-time")

accuracy = 0.73
precision = 55564 / (55564 + 20186)
recall = 55564 / (55564 + 0)
f0 = (precision * recall) / recall

axs[1][2].bar(np.array(['Accuracy', 'Precision', 'Recall', 'F(0)']), np.array([accuracy, precision, recall, f0]))
axs[1][2].set_title("Final result")

# for ax in axs.flat:
#     ax.label_outer()



plt.show()
