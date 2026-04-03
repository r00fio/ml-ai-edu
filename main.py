import math
import random


# sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# производная
def sigmoid_derivative(y):
    return y * (1 - y)


# данные AND
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# случайные веса
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

lr = 0.5

# обучение
for epoch in range(5000):
    total_loss = 0

    for inputs, target in data:
        x1, x2 = inputs

        # forward
        z = w1 * x1 + w2 * x2 + b
        y = sigmoid(z)

        # ошибка
        loss = 0.5 * (y - target) ** 2
        total_loss += loss

        # backward
        dL_dy = y - target
        dy_dz = sigmoid_derivative(y)

        dL_dw1 = dL_dy * dy_dz * x1
        dL_dw2 = dL_dy * dy_dz * x2
        dL_db = dL_dy * dy_dz

        # update
        w1 -= lr * dL_dw1
        w2 -= lr * dL_dw2
        b -= lr * dL_db

    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss {total_loss:.4f}")

print("\nFinal weights:")
print("w1 =", w1)
print("w2 =", w2)
print("b  =", b)
print([[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)])
print(range(len([3,2,2,1])))
# проверка
for inputs, _ in data:
    z = w1 * inputs[0] + w2 * inputs[1] + b
    y = sigmoid(z)
    print(inputs, round(y, 3))
