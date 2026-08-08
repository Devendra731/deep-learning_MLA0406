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

cm = confusion_matrix(actual, predicted)

sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    xticklabels=['Dog','Not Dog'],
    yticklabels=['Dog','Not Dog']
)

plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.title("Confusion Matrix")
plt.show()
