import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X, y = make_blobs(
    n_samples=600,
    centers=3,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(2, activation='linear', input_shape=(2,)),
    Dense(2, activation='linear'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Experiment 20")
print("Learning Rate: 0.01")
print("Activation: Linear")
print("Hidden Layers: 2")
print("Hidden Neurons: 2")
print("Number of Classes: 3")
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy, 4))

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multi Class Data')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Multi Class Neural Network')
plt.legend()
plt.show()
