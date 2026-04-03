import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, inputs):
        result = np.array(inputs) @ np.array(self.weights)
        result += self.bias
        result = sigmoid(result)
        return round(result, 3)



# Первый слой
or_neuron = Perceptron(weights=[10, 10], bias=-5)
and_neuron = Perceptron(weights=[10, 10], bias=-15)

# Второй слой
final_neuron = Perceptron(weights=[10, -20], bias=-5)


def xor(input):
    or_result = or_neuron.predict(input)
    and_result = and_neuron.predict(input)
    final_result = final_neuron.predict([or_result, and_result])
    return final_result


tests = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

for t in tests:
    print(t, "->", xor(t))
