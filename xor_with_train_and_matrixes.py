import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


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

lr = 0.1

for epoch in range(50000):
    # выходы скрытых слоев - матрица 4x2 - 4 теста на 2 нейрона
    h = sigmoid(X @ W1 + B1)
    # выход выходного слоя - матрица 4x1 - 4 теста на 1 нейрон
    out = sigmoid(h @ W2 + B2)

    # ошибка и дельта на выходном слое
    # матрица ошибок по результатам 4x1 (4 теста)
    error_out = Y - out
    # вычисляем на сколько быстро растет функция что бы шагнуть вперед
    d_out = error_out * sigmoid_derivative(out)

    # вычисляем ошибку скрытого слоя
    # выход: “я ошибся на 0.7”
    # скрытый нейрон:
    # “я влиял на выход через вес 2 → значит моя ошибка ≈ 1.4”
    error_h = d_out @ W2.T  # матрица 4x2 - 4 теста и 2 нейрона
    d_h = error_h * sigmoid_derivative(h)
    # обновляем веса выходного слоя
    W2 += (h.T @ d_out) * lr
    # суммируем результаты всех 4 тестов у 1 нейрона и предыдущего биаса
    B2 += np.sum(d_out, axis=0, keepdims=True) * lr
    # обновляем веса скрытых слоев
    W1 += (X.T @ d_h) * lr
    # суммируем результаты всех 4 тестов у 2 нейронов и предыдущего биаса
    B1 += np.sum(d_h, axis=0, keepdims=True) * lr

final = sigmoid(sigmoid(X @ W1 + B1) @ W2 + B2)

for i in range(4):
    print(f"Вход {X[i]} -> Результат: {final[i][0]:.4f} (Цель: {Y[i][0]})")