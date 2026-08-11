import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def mean_squared_error(y_true, y_predicted):
    return np.sum((y_true - y_predicted) ** 2) / len(y_true)

def gradient_descent(
    x,
    y,
    iterations=1000,
    learning_rate=0.01,
    stopping_threshold=1e-6
):

    weight = 0.0
    bias = 0.0

    n = float(len(x))

    costs = []

    previous_cost = None

    for i in range(iterations):

        y_predicted = weight * x + bias

        cost = mean_squared_error(
            y,
            y_predicted
        )

        if previous_cost is not None:

            if abs(previous_cost - cost) <= stopping_threshold:
                break

        previous_cost = cost

        costs.append(cost)

        weight_derivative = (
            -(2/n) * sum(
                x * (y - y_predicted)
            )
        )

        bias_derivative = (
            -(2/n) * sum(
                y - y_predicted
            )
        )

        weight -= learning_rate * weight_derivative

        bias -= learning_rate * bias_derivative

    plt.plot(costs)

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")

    plt.show()

    return weight, bias


X = np.array([
    32.5,53.4,61.5,47.4,59.8,
    55.1,52.2,39.2,48.1,52.5,
    45.4,54.3,44.1,58.1,56.7,
    48.9,44.6,60.2,45.6,38.8
])

Y = np.array([
    31.7,68.7,62.5,71.5,87.2,
    78.2,79.6,59.1,75.3,71.3,
    55.1,82.4,62.0,75.3,81.4,
    60.7,82.8,97.3,48.8,56.8
])

scaler = StandardScaler()

X_normalized = scaler.fit_transform(
    X.reshape(-1,1)
).flatten()

weight, bias = gradient_descent(
    X_normalized,
    Y,
    iterations=2000,
    learning_rate=0.01
)

print("Estimated Weight:", weight)
print("Estimated Bias:", bias)

Y_pred = weight * X_normalized + bias

plt.scatter(X, Y, label="Data Points")

plt.plot(
    X,
    Y_pred,
    linestyle="--",
    label="Fitted Line"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "Linear Regression using Gradient Descent"
)

plt.legend()

plt.show()
