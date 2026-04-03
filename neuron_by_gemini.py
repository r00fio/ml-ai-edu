import numpy as np


# 1. Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(p):
    # p — это уже результат сигмоиды
    return p * (1 - p)


# 2. Данные (X — входы, Y — идеальные ответы)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([[0], [1], [1], [0]])

# 3. Инициализация весов и смещений (Bias)
# Скрытый слой: 2 входа -> 3 нейрона
W1 = np.random.uniform(-1, 1, (2, 3))
B1 = np.random.uniform(-1, 1, (1, 3))

# Выходной слой: 3 нейрона -> 1 выход
W2 = np.random.uniform(-1, 1, (3, 1))
B2 = np.random.uniform(-1, 1, (1, 1))

lr = 0.5  # Learning Rate

# 4. Обучение
for epoch in range(40000):
    # --- FORWARD PASS (Используем @ вместо np.dot) ---
    l1 = sigmoid(X @ W1 + B1)
    l2 = sigmoid(l1 @ W2 + B2)

    # --- BACKWARD PASS ---
    # Ошибка и дельта на выходе
    error_out = Y - l2
    d_out = error_out * sigmoid_der(l2)

    # Ошибка и дельта на скрытом слое
    # .T — это транспонирование (поворот матрицы), чтобы веса "совпали" при обратном ходе
    error_h = d_out @ W2.T
    d_h = error_h * sigmoid_der(l1)

    # --- UPDATE (Градиентный спуск) ---
    W2 += (l1.T @ d_out) * lr
    B2 += np.sum(d_out, axis=0) * lr

    W1 += (X.T @ d_h) * lr
    B1 += np.sum(d_h, axis=0) * lr

# 5. Проверка
print("--- Итоговые предсказания XOR ---")
# Финальный проход (Forward) одной строкой
final = sigmoid(sigmoid(X @ W1 + B1) @ W2 + B2)

for i in range(4):
    print(f"Вход {X[i]} -> Результат: {final[i][0]:.4f} (Цель: {Y[i][0]})")