import numpy as np

lr = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

class Perceptron:
    def __init__(self, bias):
        self.weights = [0,0]
        self.bias = bias

    def train(self, inputs, y):
        result = np.array(inputs) @ np.array(self.weights)
        result += self.bias
        result = sigmoid(result)
        # backward
        dL_dy = result - y
        dy_dz = sigmoid_derivative(result)

        dL_dw1 = dL_dy * dy_dz * inputs[0]
        dL_dw2 = dL_dy * dy_dz * inputs[1]
        self.weights[0] -= lr * dL_dw1
        self.weights[1] -= lr * dL_dw2
        self.bias -= lr * dy_dz * dL_dy
        return result



# Первый слой
# or_neuron = Perceptron(weights=[10, 10], bias=-5)
and_neuron = Perceptron(bias=0)

# Второй слой
# final_neuron = Perceptron(weights=[10, -20], bias=-5)


def do_and(input, y):
    # or_result = or_neuron.predict(input)
    and_result = and_neuron.train(input, y)
    # final_result = final_neuron.predict([or_result, and_result])
    # return final_result
    return and_result


X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = np.array([[0], [0], [0], [1]])


for epoch in range(4000):
    for t in range(len(X)):
        do_and(X[t], Y[t])

for t in range(len(X)):
    print(X[t], "->", do_and(X[t], Y[t]))
