import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


# данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y = np.array([[0], [1], [1], [0]])
# скрытый слой 2 нейрона 2 входа 1 выход
W1 = np.random.uniform(-1, 1, (4, 2))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B1 = np.random.uniform(-1, 1, (1, 2))
# скрытый слой 2 нейрона 2 входа 1 выход
W0 = np.random.uniform(-1, 1, (2, 4))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B0 = np.random.uniform(-1, 1, (1, 4))

# выходной слой 1 нейрон 2 входа 1 выход
W2 = np.random.uniform(-1, 1, (2, 1))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B2 = np.random.uniform(-1, 1, (1, 1))

lr = 0.1

for epoch in range(5000):
    # выходы скрытых слоев - матрица 4x2 - 4 теста на 2 нейрона
    h0 = sigmoid(X @ W0 + B0)

    h1 = sigmoid(h0 @ W1 + B1)
    # выход выходного слоя - матрица 4x1 - 4 теста на 1 нейрон
    out = sigmoid(h1 @ W2 + B2)

    # ошибка и дельта на выходном слое
    # матрица ошибок по результатам 4x1 (4 теста)
    error_out = out - Y
    # вычисляем на сколько быстро растет функция что бы шагнуть вперед
    d_out = error_out * sigmoid_derivative(out)

    # вычисляем ошибку скрытого слоя
    # выход: “я ошибся на 0.7”
    # скрытый нейрон:
    # “я влиял на выход через вес 2 → значит моя ошибка ≈ 1.4”
    error_h1 = d_out @ W2.T  # матрица 4x2 - 4 теста и 2 нейрона
    d_h1 = error_h1 * sigmoid_derivative(h1)

    error_h0 = d_h1 @ W1.T
    d_h0 = error_h0 * sigmoid_derivative(h0)
    # обновляем веса
    W2 -= (h1.T @ d_out) * lr
    W1 -= (h0.T @ d_h1) * lr
    W0 -= (X.T @ d_h0) * lr

    B2 -= np.sum(d_out, axis=0, keepdims=True) * lr
    B1 -= np.sum(d_h1, axis=0, keepdims=True) * lr
    B0 -= np.sum(d_h0, axis=0, keepdims=True) * lr

final = sigmoid(sigmoid(sigmoid(X @ W0 + B0) @ W1 + B1) @ W2 + B2)

for i in range(4):
    print(f"Вход {X[i]} -> Результат: {final[i][0]:.4f} (Цель: {Y[i][0]})")