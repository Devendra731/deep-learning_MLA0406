import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data['species'] = iris.target

X = data[['sepal length (cm)']]
y = data['sepal width (cm)']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear Regression")

plt.legend()
plt.show()

new_sample = pd.DataFrame(
    [[5]],
    columns=['sepal length (cm)']
)

prediction = model.predict(new_sample)

print(
    "Predicted Sepal Width:",
    prediction[0]
)
