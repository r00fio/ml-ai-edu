import numpy as np

lr = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


class Perceptron:
    def __init__(self):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()

    def forward(self, inputs):
        z = np.array(inputs) @ self.weights + self.bias
        return sigmoid(z)


# сеть XOR
class XOR_Network:
    def __init__(self):
        # скрытый слой (2 нейрона)
        self.h1 = Perceptron()
        self.h2 = Perceptron()

        # выходной нейрон (принимает 2 входа)
        self.out = Perceptron()

    #
    # self.out.weights[0] = w5
    # self.out.weights[1] = w6
    # self.h1.weights[0] = w4
    # self.h1.weights[1] = w3
    # self.h2.weights[0] = w2
    # self.h2.weights[1] = w1
    #     x1 --(w1)--\
    #                 > h1 --(w5)--\
    #     x2 --(w2)--/              \
    #                                > out
    #    x1 --(w3)--\               /
    #                 > h2 --(w6)--/
    #    x2 --(w4)--/

    def train(self, inputs, y):
        # ===== FORWARD =====
        h1_out = self.h1.forward(inputs)
        h2_out = self.h2.forward(inputs)
        final_out = self.out.forward([h1_out, h2_out])

        # ===== BACKWARD =====
        # выходной слой
        # градиенты выходного слоя. Тут мы берем h1_out и h2_out так как они являются входами выходного слою.
        dL_dw_out_err = self.errorBackPropagation(final_out - y, self.out, inputs, final_out)

        # скрытый слой (вычисляем и распространяем ошибку назад)
        self.errorBackPropagation(dL_dw_out_err * self.out.weights[0], self.h1, inputs, h1_out)
        self.errorBackPropagation(dL_dw_out_err * self.out.weights[1], self.h2, inputs, h2_out)

        return final_out

    def errorBackPropagation(self, dL_dh, neuron, inputs, h_out):
        dh_dz = sigmoid_derivative(h_out)
        dL_dw = dL_dh * dh_dz
        # Вычисляем новые веса
        neuron.weights[0] -= lr * dL_dw * inputs[0]
        neuron.weights[1] -= lr * dL_dw * inputs[1]
        neuron.bias -= lr * dL_dw
        return dL_dw

    def predict(self, inputs):
        h1_out = self.h1.forward(inputs)
        h2_out = self.h2.forward(inputs)
        final_out = self.out.forward([h1_out, h2_out])
        return final_out


# данные XOR
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = np.array([0, 1, 1, 0])

# обучение
net = XOR_Network()

for epoch in range(50000):
    for i in range(len(X)):
        net.train(X[i], Y[i])

# тест
for i in range(len(X)):
    print(X[i], "->", net.predict(X[i]))
