import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

actual = np.array([
    'Dog','Dog','Dog','Not Dog','Dog',
    'Not Dog','Dog','Dog','Not Dog','Not Dog'
])

predicted = np.array([
    'Dog','Not Dog','Dog','Not Dog','Dog',
    'Dog','Dog','Dog','Not Dog','Not Dog'
])

conf_matrix = confusion_matrix(actual, predicted)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='g',
    xticklabels=['Dog','Not Dog'],
    yticklabels=['Dog','Not Dog']
)

plt.ylabel("Prediction")
plt.xlabel("Actual")
plt.title("Confusion Matrix")
plt.show()
