import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y = np.array([[0], [1], [1], [0]])
# скрытый слой 2 нейрона 2 входа 1 выход
W1 = np.random.uniform(-1, 1, (2, 2))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B1 = np.random.uniform(-1, 1, (1, 2))

# выходной слой 1 нейрон 2 входа 1 выход
W2 = np.random.uniform(-1, 1, (2, 1))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B2 = np.random.uniform(-1, 1, (1, 1))

lr = 0.01
for epoch in range(50000):

    # FORWARD
    z1 = X @ W1 + B1
    h = relu(z1)

    z2 = h @ W2 + B2
    out = sigmoid(z2)

    # BACKWARD
    error_out = out - Y
    d_out = error_out * sigmoid_derivative(out)

    error_h = d_out @ W2.T
    d_h = error_h * relu_derivative(z1)

    # UPDATE
    W2 -= (h.T @ d_out) * lr
    B2 -= np.sum(d_out, axis=0, keepdims=True) * lr

    W1 -= (X.T @ d_h) * lr
    B1 -= np.sum(d_h, axis=0, keepdims=True) * lr

final = sigmoid(sigmoid(X @ W1 + B1) @ W2 + B2)

for i in range(4):
    print(f"Вход {X[i]} -> Результат: {final[i][0]:.4f} (Цель: {Y[i][0]})")