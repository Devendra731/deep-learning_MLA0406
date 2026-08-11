from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=23
)

model = RandomForestClassifier(random_state=23)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel("Prediction")
plt.xlabel("Actual")
plt.title("Multi-Class Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
